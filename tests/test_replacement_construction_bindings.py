from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.replacement_construction_bindings import (
    GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
    MASK_SET_QUALIFICATION_SCHEMA_VERSION,
    ReplacementConstructionBindingsError,
    materialize_replacement_construction_bindings,
    seal_replacement_construction_bindings,
    validate_replacement_construction_bindings,
)
from blueprint_pipeline.simready_graph_asset_static_qualification import (
    SCHEMA_VERSION as STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
)
from blueprint_pipeline.simready_replacement_native_qualification import (
    NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION,
    materialize_simready_replacement_native_qualification,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _num_sha(value: int) -> str:
    return "sha256:" + f"{value:064x}"


def _row(suffix: str, digest_characters: str) -> dict:
    values = iter(digest_characters)
    return {
        "task_id": f"task_{suffix}",
        "asset_id": f"replacement_{suffix}",
        "task_freeze_digest": _sha(next(values)),
        "source_object_instance_id": f"source_{suffix}",
        "removal_id": f"removal_{suffix}",
        "mask_set_id": f"masks_{suffix}",
        "mask_set_receipt_digest": _sha(next(values)),
        "source_removal_receipt_digest": _sha(next(values)),
        "source_removal_qualified": True,
        "collider_deletion_id": f"collider_{suffix}",
        "source_collider_prim_path": f"/Root/source_{suffix}",
        "collider_deletion_receipt_digest": _sha(next(values)),
        "collider_deletion_qualified": True,
        "replacement_qualification_id": f"qualification_{suffix}",
        "replacement_qualification_receipt_digest": _sha(next(values)),
        "replacement_asset_sha256": _sha(next(values)),
        "replacement_simulator_import_qualified": True,
    }


def _sealed() -> dict:
    return seal_replacement_construction_bindings(
        scene_freeze_digest=_sha("1"),
        task_freeze_join_digest=_sha("2"),
        bindings=[_row("a", "345678"), _row("b", "9abcde")],
    )


def test_two_independent_removal_replacement_lanes_seal() -> None:
    sealed = _sealed()

    assert validate_replacement_construction_bindings(sealed) == sealed
    assert [row["asset_id"] for row in sealed["bindings"]] == [
        "replacement_a",
        "replacement_b",
    ]


@pytest.mark.parametrize("count", [1, 2, 5])
def test_general_construction_set_accepts_one_to_five_replacements(count: int) -> None:
    rows = [
        _row(chr(ord("a") + index), "345678")
        for index in range(count)
    ]
    # Make every evidence digest independent as required by the contract.
    for index, row in enumerate(rows):
        for offset, field in enumerate(
            (
                "task_freeze_digest",
                "mask_set_receipt_digest",
                "source_removal_receipt_digest",
                "collider_deletion_receipt_digest",
                "replacement_qualification_receipt_digest",
                "replacement_asset_sha256",
            )
        ):
            row[field] = "sha256:" + f"{index * 6 + offset + 1:064x}"

    sealed = seal_replacement_construction_bindings(
        scene_freeze_digest=_sha("1"),
        task_freeze_set_digest=_sha("2"),
        bindings=rows,
    )

    assert len(sealed["bindings"]) == count
    assert sealed["schema_version"] == "replacement_construction_bindings.v2"


def test_general_construction_set_rejects_six_replacements() -> None:
    rows = [_row(chr(ord("a") + index), "345678") for index in range(6)]

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        seal_replacement_construction_bindings(
            scene_freeze_digest=_sha("1"),
            task_freeze_set_digest=_sha("2"),
            bindings=rows,
        )

    assert "replacement_construction_binding_count_out_of_range" in excinfo.value.errors


@pytest.mark.parametrize(
    "field",
    [
        "mask_set_id",
        "source_removal_receipt_digest",
        "collider_deletion_id",
        "collider_deletion_receipt_digest",
        "replacement_qualification_id",
        "replacement_qualification_receipt_digest",
        "replacement_asset_sha256",
    ],
)
def test_shared_removal_or_replacement_identity_is_rejected(field: str) -> None:
    sealed = _sealed()
    sealed["bindings"][1][field] = sealed["bindings"][0][field]
    sealed["construction_digest"] = canonical_digest(sealed, digest_field="construction_digest")

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        validate_replacement_construction_bindings(sealed)

    assert f"replacement_construction_shared_identity:{field}" in excinfo.value.errors


def test_unqualified_receipt_and_digest_mutation_fail_closed() -> None:
    sealed = _sealed()
    sealed["bindings"][0]["source_removal_qualified"] = False

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        validate_replacement_construction_bindings(sealed)

    assert (
        "replacement_construction_qualification_missing:0:source_removal_qualified"
        in excinfo.value.errors
    )
    assert "replacement_construction_digest_invalid" in excinfo.value.errors


def test_swapped_task_freeze_binding_changes_construction_seal() -> None:
    sealed = _sealed()
    swapped = copy.deepcopy(sealed)
    swapped["bindings"][0]["task_freeze_digest"], swapped["bindings"][1]["task_freeze_digest"] = (
        swapped["bindings"][1]["task_freeze_digest"],
        swapped["bindings"][0]["task_freeze_digest"],
    )

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        validate_replacement_construction_bindings(swapped)

    assert excinfo.value.errors == ("replacement_construction_digest_invalid",)


def _write_receipt(path: Path, payload: dict) -> Path:
    payload["receipt_digest"] = canonical_digest(payload, digest_field="receipt_digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _path_backed_packet(
    tmp_path: Path,
    *,
    object_count: int = 2,
    force_batch_collider_receipt: bool = False,
) -> tuple[Path, list[dict[str, str]]]:
    root = Path(__file__).resolve().parents[1]
    manifests = root / "docs/arm_decision_proof_v1/manifests"
    scene_path = manifests / "third_scene_840920_dual_task_scene_freeze.v1.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    base_task_paths = [
        manifests / "third_scene_840920_task_a_freeze.v1.json",
        manifests / "third_scene_840920_task_b_freeze.v1.json",
    ]
    task_paths = base_task_paths[:object_count]
    base_extra = json.loads(base_task_paths[1].read_text(encoding="utf-8"))
    for index in range(2, object_count):
        suffix = chr(ord("a") + index)
        extra = copy.deepcopy(base_extra)
        extra["task_id"] = f"task_{suffix}"
        extra["source_object"]["instance_id"] = f"source_{suffix}"
        for field in (
            "removal_id",
            "mask_set_id",
            "collider_deletion_id",
            "replacement_asset_id",
            "replacement_qualification_id",
        ):
            extra["removal_plan"][field] = f"{field}_{suffix}"
        extra["removal_plan"]["source_collider_prim_path"] = f"/Root/source_{suffix}"
        for role in ("external", "wrist", "overview"):
            extra["cameras"][role] = f"{role}_{suffix}"
        extra["task_freeze_digest"] = canonical_digest(
            extra, digest_field="task_freeze_digest"
        )
        task_path = tmp_path / f"task_{suffix}.json"
        task_path.write_text(json.dumps(extra, sort_keys=True) + "\n", encoding="utf-8")
        task_paths.append(task_path)
    lanes: list[dict[str, str]] = []
    collider_rows: list[dict] = []
    for index, task_path in enumerate(task_paths):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        removal = task["removal_plan"]
        common = {
            "scene_id": scene["selected_scene_id"],
            "scene_freeze_digest": scene["scene_freeze_digest"],
            "task_id": task["task_id"],
            "task_freeze_digest": task["task_freeze_digest"],
            "source_object_instance_id": task["source_object"]["instance_id"],
            "removal_id": removal["removal_id"],
            "mask_set_id": removal["mask_set_id"],
        }
        lane_root = tmp_path / f"lane_{index}"
        mask = _write_receipt(
            lane_root / "mask.json",
            {
                "schema_version": MASK_SET_QUALIFICATION_SCHEMA_VERSION,
                "status": "calibrated_mask_set_qualified",
                **common,
                "source_scene_sha256": scene["source_components"]["interiorgs"]["sha256"],
                "calibrated_masks_qualified": True,
                "receipt_digest": "",
            },
        )
        mask_value = json.loads(mask.read_text(encoding="utf-8"))
        gaussian = _write_receipt(
            lane_root / "gaussian.json",
            {
                "schema_version": GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
                "status": "source_gaussian_removal_qualified",
                **common,
                "source_scene_sha256": scene["source_components"]["interiorgs"]["sha256"],
                "mask_set_receipt_digest": mask_value["receipt_digest"],
                "source_removal_qualified": True,
                "retained_records_byte_exact": True,
                "protected_geometry_deleted": False,
                "receipt_digest": "",
            },
        )
        collider = _write_receipt(
            lane_root / "collider.json",
            {
                "schema_version": "source_collider_subtree_removal.v1",
                "status": "exact_source_collider_subtree_removed",
                "removal_id": removal["collider_deletion_id"],
                "sage_collision_usd_sha256": scene["source_components"]["sage_collision"]["sha256"],
                "removed_prim_path": removal["source_collider_prim_path"],
                "source_bytes_unchanged": True,
                "unrelated_prim_inventory_unchanged": True,
                "remaining_target_collision_prim_count": 0,
                "removed_prim_count": 1,
                "replacement_inserted": False,
                "receipt_digest": "",
            },
        )
        collider_value = json.loads(collider.read_text(encoding="utf-8"))
        collider_rows.append(
            {
                "removal_id": removal["collider_deletion_id"],
                "target_prim_path": removal["source_collider_prim_path"],
                "source_scene_sha256": scene["source_components"]["sage_collision"]["sha256"],
                "removed_prim_count": 1,
                "removed_prim_paths_digest": _num_sha(index + 7),
                "removed_scene": {
                    "relative_path": f"lane_{index}/removed.usda",
                    "size_bytes": 1,
                    "sha256": _num_sha(index + 5),
                },
                "receipt": {
                    "relative_path": collider.relative_to(tmp_path).as_posix(),
                    "size_bytes": collider.stat().st_size,
                    "sha256": _file_sha256(collider),
                },
                "receipt_digest": collider_value["receipt_digest"],
            }
        )
        asset_sha256 = _num_sha(index + 3)
        qualification_identity = {
            key: common[key]
            for key in (
                "scene_id",
                "scene_freeze_digest",
                "task_id",
                "task_freeze_digest",
                "source_object_instance_id",
            )
        }
        static = _write_receipt(
            lane_root / "static_qualification.json",
            {
                "schema_version": STATIC_GRAPH_ASSET_QUALIFICATION_SCHEMA_VERSION,
                "status": "authored_structure_statically_qualified",
                "task_id": task["task_id"],
                "task_freeze_digest": task["task_freeze_digest"],
                "asset_id": removal["replacement_asset_id"],
                "replacement_usd": {
                    "path": "/fixture/replacement.usda",
                    "size_bytes": 123,
                    "sha256": asset_sha256,
                },
                "authored_structure_statically_qualified": True,
                "structural_findings": [],
                "contract_blockers": ["native_simulator_import_unexecuted"],
                "receipt_digest": "",
            },
        )
        native_import = _write_receipt(
            lane_root / "native_import.json",
            {
                "schema_version": NATIVE_IMPORT_RECEIPT_SCHEMA_VERSION,
                "status": "native_import_qualified",
                **qualification_identity,
                "asset_id": removal["replacement_asset_id"],
                "replacement_qualification_id": removal["replacement_qualification_id"],
                "replacement_asset_sha256": asset_sha256,
                "native_isaac_executed": True,
                "native_simulator_import_qualified": True,
                "physical_equivalence_claimed": False,
                "simulator_import_identity": {
                    "runtime": "fixture_native_import",
                    "imported_prim_path": f"/World/{removal['replacement_asset_id']}",
                },
                "receipt_digest": "",
            },
        )
        replacement = materialize_simready_replacement_native_qualification(
            scene_freeze_receipt_path=scene_path,
            task_freeze_receipt_path=task_path,
            static_qualification_receipt_path=static,
            native_import_receipt_path=native_import,
            output_path=lane_root / "replacement.json",
        )
        replacement_path = lane_root / "replacement.json"
        assert replacement["replacement_asset_sha256"] == asset_sha256
        lanes.append(
            {
                "task_freeze_receipt_path": str(task_path),
                "mask_set_receipt_path": str(mask),
                "gaussian_removal_receipt_path": str(gaussian),
                "source_collider_deletion_receipt_path": str(collider),
                "replacement_qualification_receipt_path": str(replacement_path),
            }
        )
    batch = _write_receipt(
        tmp_path / "collider_batch.json",
        {
            "schema_version": "source_collider_batch_removal.v1",
            "status": "independent_and_shared_source_colliders_removed",
            "source_scene_usd": {
                "path": "/fixture/source.usda",
                "size_bytes": 100,
                "sha256": scene["source_components"]["sage_collision"]["sha256"],
            },
            "source_bytes_unchanged": True,
            "unrelated_prim_inventory_unchanged": True,
            "remaining_target_collision_prim_count": 0,
            "replacement_inserted": False,
            "independent_receipts_share_exact_source_digest": True,
            "independent_removed_scenes_are_distinct": True,
            "target_count": len(collider_rows),
            "target_removals": collider_rows,
            "receipt_digest": "",
        },
    )
    if object_count > 1 or force_batch_collider_receipt:
        for lane in lanes:
            lane["source_collider_deletion_receipt_path"] = str(batch)
    return scene_path, lanes


def test_path_backed_materializer_derives_all_claims_from_receipts(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path)
    output = tmp_path / "result" / "construction.json"

    result = materialize_replacement_construction_bindings(
        scene_freeze_receipt_path=scene_path,
        evidence_lanes=lanes,
        output_path=output,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert len(result["bindings"]) == 2
    assert all(row["source_removal_qualified"] for row in result["bindings"])
    assert all(row["collider_deletion_qualified"] for row in result["bindings"])
    assert all(row["replacement_simulator_import_qualified"] for row in result["bindings"])
    assert all(
        set(row["evidence_receipts"])
        == {
            "task_freeze",
            "mask_set",
            "gaussian_removal",
            "source_collider_deletion",
            "replacement_qualification",
        }
        for row in result["bindings"]
    )


def test_path_backed_materializer_supports_five_independent_objects(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path, object_count=5)

    result = materialize_replacement_construction_bindings(
        scene_freeze_receipt_path=scene_path,
        evidence_lanes=lanes,
    )

    assert len(result["bindings"]) == 5
    assert len({row["asset_id"] for row in result["bindings"]}) == 5
    assert len({row["source_removal_receipt_digest"] for row in result["bindings"]}) == 5


def test_path_backed_materializer_supports_one_independent_object(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path, object_count=1)

    result = materialize_replacement_construction_bindings(
        scene_freeze_receipt_path=scene_path,
        evidence_lanes=lanes,
    )

    assert len(result["bindings"]) == 1


def test_path_backed_materializer_supports_one_object_batch_collider_receipt(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(
        tmp_path,
        object_count=1,
        force_batch_collider_receipt=True,
    )

    result = materialize_replacement_construction_bindings(
        scene_freeze_receipt_path=scene_path,
        evidence_lanes=lanes,
    )

    assert len(result["bindings"]) == 1
    evidence = result["bindings"][0]["evidence_receipts"]["source_collider_deletion"]
    task = json.loads(Path(lanes[0]["task_freeze_receipt_path"]).read_text())
    assert evidence["batch"]["sha256"].startswith("sha256:")
    assert evidence["selected_deletion_id"] == task["removal_plan"]["collider_deletion_id"]


@pytest.mark.parametrize(
    ("path_field", "expected_error"),
    [
        ("mask_set_receipt_path", "mask_set:1_identity_mismatch"),
        (
            "replacement_qualification_receipt_path",
            "replacement_qualification:1_identity_mismatch",
        ),
    ],
)
def test_path_backed_materializer_rejects_shared_or_swapped_task_evidence(
    tmp_path: Path, path_field: str, expected_error: str
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path)
    lanes[1][path_field] = lanes[0][path_field]

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        materialize_replacement_construction_bindings(
            scene_freeze_receipt_path=scene_path,
            evidence_lanes=lanes,
        )

    assert any(expected_error in error for error in excinfo.value.errors)


def test_path_backed_materializer_rejects_replacement_without_native_evidence(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path)
    replacement_path = Path(lanes[0]["replacement_qualification_receipt_path"])
    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    replacement.pop("evidence_receipts")
    replacement["receipt_digest"] = canonical_digest(
        replacement,
        digest_field="receipt_digest",
    )
    replacement_path.write_text(json.dumps(replacement, sort_keys=True) + "\n")

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        materialize_replacement_construction_bindings(
            scene_freeze_receipt_path=scene_path,
            evidence_lanes=lanes,
        )

    assert any(
        "replacement_qualification:0_native_evidence_missing" in error
        for error in excinfo.value.errors
    )


def test_path_backed_materializer_rejects_swapped_collider_deletion_ids(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path)
    batch_path = Path(lanes[0]["source_collider_deletion_receipt_path"])
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    first, second = batch["target_removals"]
    first["removal_id"], second["removal_id"] = (
        second["removal_id"],
        first["removal_id"],
    )
    _write_receipt(batch_path, batch)

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        materialize_replacement_construction_bindings(
            scene_freeze_receipt_path=scene_path,
            evidence_lanes=lanes,
        )

    assert any(
        "source_collider_deletion:0_identity_mismatch" in error for error in excinfo.value.errors
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_removal_qualified", True),
        ("source_removal_receipt_digest", _sha("f")),
        ("replacement_simulator_import_qualified", True),
    ],
)
def test_path_backed_materializer_rejects_caller_authored_claims(
    tmp_path: Path, field: str, value: object
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path)
    lanes[0][field] = value  # type: ignore[assignment]

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        materialize_replacement_construction_bindings(
            scene_freeze_receipt_path=scene_path,
            evidence_lanes=lanes,
        )

    assert "replacement_construction_lane_paths_invalid:0" in excinfo.value.errors


def test_path_backed_materializer_rejects_stale_hand_authored_receipt_digest(
    tmp_path: Path,
) -> None:
    scene_path, lanes = _path_backed_packet(tmp_path)
    gaussian_path = Path(lanes[0]["gaussian_removal_receipt_path"])
    gaussian = json.loads(gaussian_path.read_text(encoding="utf-8"))
    gaussian["source_removal_qualified"] = False
    gaussian_path.write_text(json.dumps(gaussian), encoding="utf-8")

    with pytest.raises(ReplacementConstructionBindingsError) as excinfo:
        materialize_replacement_construction_bindings(
            scene_freeze_receipt_path=scene_path,
            evidence_lanes=lanes,
        )

    assert "replacement_construction_gaussian_removal:0_digest_invalid" in (excinfo.value.errors)
