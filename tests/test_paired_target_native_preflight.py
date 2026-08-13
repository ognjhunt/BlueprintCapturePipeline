from __future__ import annotations

import gzip
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_preflight import (
    PairedTargetNativePreflightError,
    materialize_paired_target_native_preflight,
)


def _write(path: Path, value: dict, field: str) -> Path:
    value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(root: Path, task_id: str) -> dict[str, str]:
    task = root / task_id
    task.mkdir(parents=True)
    usdz = task / "scene.usdz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(b"model")
    with usdz.open("wb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, body in (
                ("default.usda", b"default"),
                ("repaired_scene.nurec", buffer.getvalue()),
                ("gauss.usda", b"gauss"),
            ):
                info = zipfile.ZipInfo(name)
                padding = (-(raw.tell() + 30 + len(name.encode()))) % 64
                if padding:
                    if padding < 4:
                        padding += 64
                    info.extra = struct.pack("<HH", 0x1986, padding - 4) + b"\0" * (
                        padding - 4
                    )
                archive.writestr(info, body)
    from blueprint_pipeline.paired_target_native_preflight import _record

    members = []
    with usdz.open("rb") as raw, zipfile.ZipFile(usdz) as archive:
        for info in archive.infolist():
            raw.seek(info.header_offset)
            fields = struct.unpack("<IHHHHHIIIHH", raw.read(30))
            offset = info.header_offset + 30 + fields[-2] + fields[-1]
            members.append(
                {
                    "filename": info.filename,
                    "size_bytes": info.file_size,
                    "data_offset_bytes": offset,
                    "sha256": _record_bytes(archive.read(info)),
                }
            )
    appearance = {
        "schema_version": "public_scene_artifixer3d_native_appearance_export.v1",
        "native_import_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "isaac_nurec_usdz": _record(usdz),
        "isaac_nurec_usdz_archive_contract": {
            "compression": "stored",
            "payload_alignment_bytes": 64,
            "all_payload_offsets_aligned": True,
            "nurec_gzip_mtime_normalized_to_zero": True,
            "members": members,
        },
    }
    appearance_path = _write(task / "appearance.json", appearance, "export_digest")
    trajectory = task / "review_transforms.json"
    trajectory.write_text("{}", encoding="utf-8")
    camera_index = task / "camera_index.json"
    camera_value = {
        "camera_index_digest": "sha256:" + "1" * 64,
        "frames": [{"camera_id": f"camera_{i}"} for i in range(8)],
    }
    camera_index.write_text(json.dumps(camera_value), encoding="utf-8")
    dual = {
        "schema_version": "public_scene_artifixer3d_dual_target_inputs.v1",
        "publisher_scene_id": "840920",
        "selected_task_ids": [task_id],
        "tasks": [
            {
                "task_id": task_id,
                "scene_directory": str(task),
                "camera_count": 8,
                "physical_camera_count": 8,
                "review_trajectory": {
                    "relative_path": trajectory.name,
                    "size_bytes": trajectory.stat().st_size,
                    "sha256": _record(trajectory)["sha256"],
                },
                "camera_index": {
                    "relative_path": camera_index.name,
                    "size_bytes": camera_index.stat().st_size,
                    "sha256": _record(camera_index)["sha256"],
                },
            }
        ],
    }
    dual_path = _write(task / "dual.json", dual, "receipt_digest")
    usd = task / "replacement.usda"
    usd.write_text("#usda 1.0", encoding="utf-8")
    cad = {
        "schema_version": "simready_graph_asset_receipt.v1",
        "status": "simready_candidate_authored",
        "task_id": task_id,
        "asset_id": f"asset_{task_id}",
        "claim_boundary": {"native_simulator_import_qualified": False},
        "output_usd": {**_record(usd)},
    }
    cad_path = _write(task / "cad.json", cad, "receipt_digest")
    static = {
        "schema_version": "simready_graph_asset_static_qualification.v1",
        "task_id": task_id,
        "asset_id": cad["asset_id"],
        "authored_structure_statically_qualified": True,
        "replacement_usd": {"sha256": cad["output_usd"]["sha256"]},
    }
    static_path = _write(task / "static.json", static, "receipt_digest")
    scenario = {
        "schema_version": "third_scene_task_scenario_suite.v1",
        "scene_id": "840920",
        "task_id": task_id,
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "required_controls": ["zero_action_negative", "scripted_positive"],
        "initial_execution_order": [f"{task_id}_canonical", f"{task_id}_camera"],
    }
    scenario_path = _write(task / "scenario.json", scenario, "suite_digest")
    return {
        "task_id": task_id,
        "appearance_export_receipt_path": str(appearance_path),
        "dual_target_inputs_receipt_path": str(dual_path),
        "simready_asset_receipt_path": str(cad_path),
        "simready_static_qualification_path": str(static_path),
        "scenario_suite_path": str(scenario_path),
    }


def _record_bytes(value: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_preflight_binds_two_tasks_and_preserves_proof_boundary(tmp_path: Path) -> None:
    tasks = [_fixture(tmp_path, "task_a"), _fixture(tmp_path, "task_b")]
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    result = materialize_paired_target_native_preflight(
        scene_id="840920",
        task_records=tasks,
        collision_scene_path=collision,
        output_path=tmp_path / "result.json",
    )
    assert result["replacement_object_count"] == 2
    assert result["maximum_replacement_objects"] == 5
    assert result["native_isaac_import_executed"] is False
    assert result["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert all(len(task["camera_index"]["camera_ids"]) == 8 for task in result["tasks"])
    assert canonical_digest(result, digest_field="receipt_digest") == result["receipt_digest"]


def test_preflight_rejects_tampered_bytes_and_six_tasks(tmp_path: Path) -> None:
    task = _fixture(tmp_path, "task_a")
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    Path(task["simready_asset_receipt_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(PairedTargetNativePreflightError):
        materialize_paired_target_native_preflight(
            scene_id="840920",
            task_records=[task],
            collision_scene_path=collision,
            output_path=tmp_path / "result.json",
        )
    with pytest.raises(PairedTargetNativePreflightError, match="task_count"):
        materialize_paired_target_native_preflight(
            scene_id="840920",
            task_records=[task] * 6,
            collision_scene_path=collision,
            output_path=tmp_path / "other.json",
        )
