from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_import_bundle import (
    PairedTargetNativeImportBundleError,
    build_paired_target_native_import_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source(tmp_path: Path, *, count: int) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    assets = []
    for index in range(count):
        path = tmp_path / f"asset_{index}.usda"
        path.write_text(
            '#usda 1.0\n(defaultPrim="Asset")\ndef Xform "Asset" {}\n',
            encoding="utf-8",
        )
        static_path = tmp_path / f"static_{index}.json"
        static = {
            "schema_version": "simready_graph_asset_static_qualification.v1",
            "authored_structure_statically_qualified": True,
            "replacement_usd": _record(path),
            "receipt_digest": "",
        }
        static["receipt_digest"] = canonical_digest(
            static, digest_field="receipt_digest"
        )
        static_path.write_text(json.dumps(static), encoding="utf-8")
        assets.append(
            {
                "task_id": f"task_{index}",
                "asset_id": f"asset_{index}",
                "visual_usd": _record(path),
                "asset_frame_registration": {"registration_digest": "sha256:" + str(index) * 64},
                "registered_static_qualification": {
                    **_record(static_path),
                    "receipt_digest": static["receipt_digest"],
                },
            }
        )
    tasks = []
    for subject in range(count):
        tasks.append(
            {
                "task_id": f"task_{subject}",
                "co_present_replacements": [
                    {
                        **row,
                        "task_subject": index == subject,
                        "passive_co_present": index != subject,
                    }
                    for index, row in enumerate(assets)
                ],
            }
        )
    value = {
        "schema_version": "paired_target_native_render_request.v1",
        "status": "native_render_requests_materialized_pending_isaac_execution",
        "scene_id": "fixture_scene",
        "replacement_object_count": count,
        "tasks": tasks,
        "native_isaac_executed": False,
        "provider_allocation_performed": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    path = tmp_path / "paired_target_native_render_request.v1.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("count", [1, 2, 5])
def test_builds_one_deterministic_co_present_import_bundle(tmp_path: Path, count: int) -> None:
    source = _source(tmp_path, count=count)
    result = build_paired_target_native_import_bundle(
        native_render_request_path=source,
        runner_path=ROOT / "scripts/run_paired_target_native_import_probe.py",
        output_root=tmp_path / "bundle",
        implementation_commit="a" * 40,
    )

    assert result["replacement_count"] == count
    assert result["provider_allocation_performed"] is False
    assert result["paid_execution_authorized_by_bundle"] is False
    assert result["candidate_policy_queried"] is False
    assert result["raw_nonredistributable_bytes_included"] is False
    assert result["canonical_interiorgs_included_or_mutated"] is False
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")
    with zipfile.ZipFile(result["bundle_path"]) as archive:
        names = archive.namelist()
        assert len([name for name in names if "/assets/replacement_" in name]) == count
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert "provider_runtime/paired_target_native_import_request.v1.json" in names
        request = json.loads(
            archive.read("provider_runtime/paired_target_native_import_request.v1.json")
        )
        assert request["replacement_count"] == count
        assert request["request_digest"] == canonical_digest(request, digest_field="request_digest")


def test_rejects_changed_asset_and_inconsistent_co_present_set(tmp_path: Path) -> None:
    source = _source(tmp_path, count=2)
    value = json.loads(source.read_text())
    value["tasks"][1]["co_present_replacements"].reverse()
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeImportBundleError,
        match="replacement_set_mismatch",
    ):
        build_paired_target_native_import_bundle(
            native_render_request_path=source,
            runner_path=ROOT / "scripts/run_paired_target_native_import_probe.py",
            output_root=tmp_path / "bad-set",
            implementation_commit="b" * 40,
        )

    source = _source(tmp_path / "tamper", count=1)
    value = json.loads(source.read_text())
    Path(value["tasks"][0]["co_present_replacements"][0]["visual_usd"]["path"]).write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(
        PairedTargetNativeImportBundleError,
        match="replacement_asset_invalid",
    ):
        build_paired_target_native_import_bundle(
            native_render_request_path=source,
            runner_path=ROOT / "scripts/run_paired_target_native_import_probe.py",
            output_root=tmp_path / "bad-asset",
            implementation_commit="c" * 40,
        )


def test_rejects_output_reuse_without_overwrite(tmp_path: Path) -> None:
    source = _source(tmp_path, count=1)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("owned")
    with pytest.raises(PairedTargetNativeImportBundleError, match="output_exists"):
        build_paired_target_native_import_bundle(
            native_render_request_path=source,
            runner_path=ROOT / "scripts/run_paired_target_native_import_probe.py",
            output_root=output,
            implementation_commit="d" * 40,
        )
    assert sentinel.read_text() == "owned"


def test_rejects_missing_asset_frame_registration(tmp_path: Path) -> None:
    source = _source(tmp_path, count=1)
    value = json.loads(source.read_text())
    value["tasks"][0]["co_present_replacements"][0].pop("asset_frame_registration")
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    source.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        PairedTargetNativeImportBundleError,
        match="asset_frame_registration_invalid",
    ):
        build_paired_target_native_import_bundle(
            native_render_request_path=source,
            runner_path=ROOT / "scripts/run_paired_target_native_import_probe.py",
            output_root=tmp_path / "bad-registration",
            implementation_commit="e" * 40,
        )
