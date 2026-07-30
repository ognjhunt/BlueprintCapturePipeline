from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.nvidia_warehouse_workcell import (
    PROVENANCE_FILES,
    SPRAYCAN_USD,
    TABLE_USD,
    WORKCELL_USD,
    build_native_camera_canary_spec,
    materialize_pinned_workcell,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture_material() -> tuple[dict[str, bytes], dict[str, list[str]]]:
    table = TABLE_USD
    workcell_child = "Props/general/workcell_child/child.usd"
    table_mdl = "Props/general/SM_HeavyDutyPackingTable_C02_01/materials/Wood.mdl"
    table_texture = (
        "Props/general/SM_HeavyDutyPackingTable_C02_01/materials/textures/wood.png"
    )
    spraycan_texture_1001 = (
        "Props/general/HandManipulation/paint_container_spraycan_a/Textures/albedo.1001.png"
    )
    spraycan_texture_1002 = (
        "Props/general/HandManipulation/paint_container_spraycan_a/Textures/albedo.1002.png"
    )
    material = {
        PROVENANCE_FILES[0]: b"root-usd",
        PROVENANCE_FILES[1]: b"sorting-usd",
        WORKCELL_USD: b"workcell-usd",
        workcell_child: b"workcell-child-usd",
        table: b"table-usd",
        SPRAYCAN_USD: b"spraycan-usd",
        table_mdl: b'texture_2d("./textures/wood.png")',
        table_texture: b"table-texture",
        spraycan_texture_1001: b"spraycan-texture-1001",
        spraycan_texture_1002: b"spraycan-texture-1002",
    }
    dependencies = {
        WORKCELL_USD: [
            "../../general/workcell_child/child.usd",
            "../../general/SM_HeavyDutyPackingTable_C02_01/"
            "SM_HeavyDutyPackingTable_C02_01_physics.usd",
        ],
        workcell_child: [],
        table: ["omniverse://simready.ov.nvidia.com/materials/table.usda"],
        SPRAYCAN_USD: ["omniverse://simready.ov.nvidia.com/materials/paint.usda"],
    }
    return material, dependencies


def test_materializes_hash_bound_dataset_local_closure_and_records_external_refs(
    tmp_path: Path,
) -> None:
    material, dependencies = _fixture_material()
    pinned = {path: _sha(value) for path, value in material.items() if path != "Props/general/workcell_child/child.usd"}

    def fetch(relative: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(material[relative])

    result = materialize_pinned_workcell(
        output_root=tmp_path / "assets",
        fetcher=fetch,
        dependency_reader=lambda path: dependencies[path.relative_to(tmp_path / "assets").as_posix()],
        asset_dependency_reader=lambda path: {
            TABLE_USD: ["./materials/Wood.mdl", "OmniPBR.mdl"],
            SPRAYCAN_USD: ["./Textures/albedo.<UDIM>.png"],
        }.get(path.relative_to(tmp_path / "assets").as_posix(), []),
        dependency_expander=lambda relative: (
            [
                "Props/general/HandManipulation/paint_container_spraycan_a/Textures/albedo.1001.png",
                "Props/general/HandManipulation/paint_container_spraycan_a/Textures/albedo.1002.png",
            ]
            if "<UDIM>" in relative
            else [relative]
        ),
        pinned_sha256=pinned,
        max_materialized_bytes=1024 * 1024,
    )

    assert result["status"] == "completed"
    assert result["whole_warehouse_materialized"] is False
    assert result["dataset_local_dependency_closure_complete"] is True
    assert result["file_count"] == len(material)
    assert len(result["external_dependencies"]) == 3
    assert "OmniPBR.mdl" in result["external_dependencies"]
    assert result["dependency_contract"] == {
        "usd_composition_dependencies_included": True,
        "usd_authored_asset_fields_included": True,
        "dataset_local_mdl_texture_literals_included": True,
        "udim_patterns_expanded_against_pinned_revision": True,
    }
    assert result["claim_boundary"]["policy_wam_loop_proven"] is False

    spec_path = tmp_path / "native_camera_canary_spec.json"
    spec = build_native_camera_canary_spec(
        materialization_manifest_path=tmp_path / "assets" / "materialization_manifest.json",
        output_path=spec_path,
    )
    assert spec["paid_gpu_execution_admitted"] is False
    assert spec["cameras"]["wrist"]["inherits_parent_transform"] is True
    assert "at_least_two_policy_calls_separated_by_one_wam_generated_observation" in spec[
        "required_checks"
    ]
    assert spec_path.is_file()


def test_materialization_rejects_pinned_asset_mutation(tmp_path: Path) -> None:
    material, dependencies = _fixture_material()

    def fetch(relative: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(material[relative])

    with pytest.raises(ValueError, match="pinned_sha256_mismatch"):
        materialize_pinned_workcell(
            output_root=tmp_path / "assets",
            fetcher=fetch,
            dependency_reader=lambda path: dependencies.get(
                path.relative_to(tmp_path / "assets").as_posix(), []
            ),
            pinned_sha256={PROVENANCE_FILES[0]: "0" * 64},
        )


def test_native_canary_spec_rejects_manifest_tampering(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset_revision": "tampered",
                "manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest_sha256_invalid"):
        build_native_camera_canary_spec(
            materialization_manifest_path=manifest,
            output_path=tmp_path / "spec.json",
        )
