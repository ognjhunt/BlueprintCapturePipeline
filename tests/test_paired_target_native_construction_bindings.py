from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_construction_bindings import (
    PairedTargetNativeConstructionBindingsError,
    materialize_paired_target_native_construction_bindings,
    validate_paired_target_native_construction_bindings,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict, *, digest_field: str) -> dict:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        digest_field: value[digest_field],
    }


def _source_collider_batch(root: Path, *, count: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "collision.usda"
    source.write_text('#usda 1.0\ndef Xform "source_scene" {}\n', encoding="utf-8")
    removed = root / "collider_removal" / "scene_without_source_colliders.usda"
    removed.parent.mkdir(parents=True, exist_ok=True)
    removed.write_text('#usda 1.0\ndef Xform "retained_scene" {}\n', encoding="utf-8")
    removals = []
    for index in range(count):
        removal_id = f"remove_source_{index}"
        target_prim_path = f"/Root/source_{index}"
        child = _write(
            removed.parent / "independent" / f"{removal_id}.receipt.json",
            {
                "schema_version": "source_collider_subtree_removal.v1",
                "status": "exact_source_collider_subtree_removed",
                "removal_id": removal_id,
                "sage_collision_usd_sha256": _sha(source),
                "removed_prim_path": target_prim_path,
                "removed_prim_count": 1,
                "source_bytes_unchanged": True,
                "unrelated_prim_inventory_unchanged": True,
                "remaining_target_collision_prim_count": 0,
                "replacement_inserted": False,
                "receipt_digest": "",
            },
            digest_field="receipt_digest",
        )
        child_path = Path(child["path"])
        removals.append(
            {
                "removal_id": removal_id,
                "target_prim_path": target_prim_path,
                "source_scene_sha256": _sha(source),
                "removed_prim_count": 1,
                "receipt_digest": child["receipt_digest"],
                "receipt": {
                    "relative_path": child_path.relative_to(removed.parent).as_posix(),
                    "size_bytes": child_path.stat().st_size,
                    "sha256": _sha(child_path),
                },
            }
        )
    batch = {
        "schema_version": "source_collider_batch_removal.v1",
        "status": "independent_and_shared_source_colliders_removed",
        "source_scene_usd": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha(source),
        },
        "shared_removed_scene_usd": {
            "relative_path": removed.name,
            "size_bytes": removed.stat().st_size,
            "sha256": _sha(removed),
        },
        "target_count": count,
        "target_removals": removals,
        "source_bytes_unchanged": True,
        "unrelated_prim_inventory_unchanged": True,
        "remaining_target_collision_prim_count": 0,
        "replacement_inserted": False,
        "independent_receipts_share_exact_source_digest": True,
        "independent_removed_scenes_are_distinct": True,
        "receipt_digest": "",
    }
    return Path(
        _write(
            removed.parent / "source_collider_batch_removal.v1.json",
            batch,
            digest_field="receipt_digest",
        )["path"]
    )


def _fixture(root: Path, *, count: int) -> Path:
    batch = _source_collider_batch(root, count=count)
    source_collision = root / "collision.usda"
    preflight_tasks = []
    manipulation_tasks = []
    import_rows = []
    for index in range(count):
        task_id = f"task_{index}"
        asset_id = f"asset_{index}"
        task_record = _write(
            root / task_id / "freeze.json",
            {
                "schema_version": "dual_task_task_freeze.v1",
                "task_id": task_id,
                "task_freeze_digest": "",
            },
            digest_field="task_freeze_digest",
        )
        usd = root / task_id / "registered.usda"
        usd.write_text(f'#usda 1.0\ndef Xform "asset_{index}" {{}}\n')
        usd_record = {
            "path": str(usd),
            "size_bytes": usd.stat().st_size,
            "sha256": _sha(usd),
        }
        registered_record = _write(
            root / task_id / "registered.json",
            {
                "schema_version": "registered_replacement_asset.v1",
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_digest": task_record["task_freeze_digest"],
                "output_usd": usd_record,
                "receipt_digest": "",
            },
            digest_field="receipt_digest",
        )
        probe_record = _write(
            root / "native" / "probes" / f"{index}.json",
            {
                "schema_version": "simready_replacement_native_import_probe_result.v1",
                "status": "completed",
                "asset_id": asset_id,
                "native_simulator_import_qualified": True,
                "candidate_policy_queried": False,
                "result_digest": "",
            },
            digest_field="result_digest",
        )
        preflight_tasks.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "registered_replacement_asset_receipt": registered_record,
                "registered_replacement_usd": usd_record,
            }
        )
        manipulation_tasks.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze": task_record,
                "native_construction_binding_ready": True,
                "native_task_arena_request": None,
                "pending_requirements": [
                    "native_task_arena_packet_request_missing"
                ],
            }
        )
        import_rows.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "blockers": [],
                "native_simulator_import_qualified": True,
                "probe_result_path": f"probes/{index}.json",
                "probe_result_sha256": probe_record["sha256"],
                "probe_result_digest": probe_record["result_digest"],
            }
        )
    paired_record = _write(
        root / "paired.json",
        {
            "schema_version": "paired_target_native_preflight.v1",
            "scene_id": "scene_fixture",
            "replacement_object_count": count,
            "collision_scene": {
                "path": str(source_collision),
                "size_bytes": source_collision.stat().st_size,
                "sha256": _sha(source_collision),
            },
            "tasks": preflight_tasks,
            "receipt_digest": "",
        },
        digest_field="receipt_digest",
    )
    import_record = _write(
        root / "native" / "result.json",
        {
            "schema_version": "paired_target_native_import_runtime_result.v1",
            "status": "completed",
            "scene_id": "scene_fixture",
            "replacement_count": count,
            "all_replacements_import_qualified": True,
            "candidate_policy_queried": False,
            "replacements": import_rows,
            "result_digest": "",
        },
        digest_field="result_digest",
    )
    manipulation_record = _write(
        root / "manipulation.json",
        {
            "schema_version": "paired_target_native_manipulation_preflight.v1",
            "status": "ready_for_native_construction_bindings",
            "preflight_phase": "pre_arena",
            "scene_id": "scene_fixture",
            "replacement_object_count": count,
            "task_freeze_set_digest": "sha256:" + "b" * 64,
            "paired_target_preflight": paired_record,
            "native_import_result": import_record,
            "tasks": manipulation_tasks,
            "native_import_qualified": True,
            "native_construction_bindings_ready": True,
            "native_reachability_executed": False,
            "controls_executed": False,
            "learned_policies_executed": False,
            "blockers": [],
            "pending_requirements": sorted(
                f"task_{index}:native_task_arena_packet_request_missing"
                for index in range(count)
            ),
            "receipt_digest": "",
        },
        digest_field="receipt_digest",
    )
    assert batch.is_file()
    return Path(manipulation_record["path"])


def _batch_for(manipulation_path: Path) -> Path:
    return manipulation_path.parent / "collider_removal" / "source_collider_batch_removal.v1.json"


@pytest.mark.parametrize("count", [1, 2, 5])
def test_materializes_one_to_five_path_backed_bindings(
    tmp_path: Path, count: int
) -> None:
    source = _fixture(tmp_path / "evidence", count=count)
    output = tmp_path / "binding.json"

    result = materialize_paired_target_native_construction_bindings(
        manipulation_preflight_path=source,
        source_collider_batch_removal_path=_batch_for(source),
        output_path=output,
    )

    assert validate_paired_target_native_construction_bindings(result) == result
    assert result["replacement_object_count"] == count
    assert len(result["bindings"]) == count
    assert result["native_reachability_qualified"] is False
    assert result["controls_executed"] is False
    assert result["collision_scene"]["path"].endswith(
        "collider_removal/scene_without_source_colliders.usda"
    )
    assert result["source_collider_batch_removal"]["canonical_digest"].startswith(
        "sha256:"
    )


def test_rejects_tampered_native_import_probe(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "evidence", count=2)
    probe = tmp_path / "evidence" / "native" / "probes" / "0.json"
    probe.write_text(probe.read_text() + "\n", encoding="utf-8")

    with pytest.raises(
        PairedTargetNativeConstructionBindingsError,
        match="native_import_probe_invalid",
    ):
        materialize_paired_target_native_construction_bindings(
            manipulation_preflight_path=source,
            source_collider_batch_removal_path=_batch_for(source),
            output_path=tmp_path / "binding.json",
        )


def test_rejects_qualified_boundary_claim_tamper(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "evidence", count=1)
    result = materialize_paired_target_native_construction_bindings(
        manipulation_preflight_path=source,
        source_collider_batch_removal_path=_batch_for(source),
        output_path=tmp_path / "binding.json",
    )
    result["native_reachability_qualified"] = True
    result["construction_digest"] = canonical_digest(
        result, digest_field="construction_digest"
    )

    with pytest.raises(
        PairedTargetNativeConstructionBindingsError,
        match="boundary_invalid:native_reachability_qualified",
    ):
        validate_paired_target_native_construction_bindings(result)


def test_refuses_the_original_collision_scene_as_the_shared_removed_scene(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path / "evidence", count=2)
    batch_path = _batch_for(source)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    original = Path(batch["source_scene_usd"]["path"])
    disguised_original = batch_path.parent / "original_collision.usda"
    disguised_original.write_bytes(original.read_bytes())
    batch["shared_removed_scene_usd"] = {
        "relative_path": disguised_original.name,
        "size_bytes": disguised_original.stat().st_size,
        "sha256": _sha(disguised_original),
    }
    batch["receipt_digest"] = canonical_digest(batch, digest_field="receipt_digest")
    batch_path.write_text(json.dumps(batch, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        PairedTargetNativeConstructionBindingsError,
        match="shared_scene_not_removed",
    ):
        materialize_paired_target_native_construction_bindings(
            manipulation_preflight_path=source,
            source_collider_batch_removal_path=batch_path,
            output_path=tmp_path / "must_not_exist.json",
        )


def test_refuses_tampered_shared_removed_scene_bytes(tmp_path: Path) -> None:
    source = _fixture(tmp_path / "evidence", count=2)
    batch_path = _batch_for(source)
    removed = batch_path.parent / "scene_without_source_colliders.usda"
    removed.write_text(removed.read_text(encoding="utf-8") + "# tampered\n")

    with pytest.raises(
        PairedTargetNativeConstructionBindingsError,
        match="shared_scene_invalid",
    ):
        materialize_paired_target_native_construction_bindings(
            manipulation_preflight_path=source,
            source_collider_batch_removal_path=batch_path,
            output_path=tmp_path / "must_not_exist.json",
        )
