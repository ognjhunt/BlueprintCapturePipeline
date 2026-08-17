from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from copy import deepcopy

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.fresh_scene_supervisor_bindings import (
    compile_fresh_scene_supervisor_bindings,
    materialize_fresh_scene_semantic_teacher_handoff_request,
    materialize_fresh_scene_supervisor_bindings,
)
from blueprint_pipeline.fresh_scene_paired_target_preparation import (
    STAGE_CONTRACTS,
    materialize_fresh_scene_preparation_status,
)
from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
)
from blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs import (
    HANDOFF_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SEMANTIC_TEACHER_SCHEMA,
    DualTargetInputError,
    materialize_semantic_teacher_artifixer_handoff,
)
from blueprint_pipeline.semantic_teacher_image_edit_worker import (
    RUNTIME_REQUEST_SCHEMA_VERSION,
    RUNTIME_RESULT_SCHEMA_VERSION,
)
from tests.test_public_scene_artifixer3d_candidate_inputs import _preflight
from tests.test_public_scene_calibrated_object_masks import _task


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    value["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(path: Path, value: dict[str, object], field: str) -> dict[str, object]:
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    _write(path, value)
    return value


def _fixture(tmp_path: Path) -> dict[str, object]:
    preflight = _preflight(tmp_path / "source_fixture", count=2)
    preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
    expanded = []
    for row in preflight_value["camera_inputs"]:
        for camera_index in range(8):
            duplicate = deepcopy(row)
            duplicate["camera_id"] = f"{row['camera_id']}_{camera_index}"
            duplicate["calibration"]["spec"]["pose"]["T_world_camera_opencv"][0][3] += (
                camera_index * 0.25
            )
            expanded.append(duplicate)
    preflight_value["camera_inputs"] = expanded
    preflight_value["preflight_digest"] = canonical_digest(
        preflight_value, digest_field="preflight_digest"
    )
    _write(preflight, preflight_value)
    candidate_root = tmp_path / "candidate"
    candidate = materialize_artifixer3d_candidate_inputs(
        calibrated_residual_preflight_path=preflight,
        output_root=candidate_root,
    )
    candidate_path = candidate_root / "public_scene_artifixer3d_candidate_inputs.v3.json"
    backend_entry = {
        "backend_id": "fixture_semantic_teacher",
        "capability": "semantic_teacher_image_edit",
        "execution": {
            "adapter_id": "fixture_image_edits_v1",
            "model_snapshot": "fixture-model-snapshot",
        },
    }
    backend_digest = canonical_digest(backend_entry)
    packet_tasks = [
        {
            "task_id": task["task_id"],
            "camera_count": task["camera_count"],
            "frames": [
                {
                    "frame_index": frame["frame_index"],
                    "camera_id": frame["camera_id"],
                }
                for frame in task["frames"]
            ],
        }
        for task in candidate["tasks"]
    ]
    packet_path = tmp_path / "packet.json"
    packet = _sealed(
        packet_path,
        {
            "schema_version": "fresh_scene_semantic_teacher_image_edit_packet.v1",
            "status": ("semantic_teacher_image_edit_packet_prepared_no_upload_no_execution"),
            "source_candidate_inputs_receipt": {
                **_record(candidate_path),
                "receipt_digest": candidate["receipt_digest"],
            },
            "backend": {
                "registry_entry": backend_entry,
                "backend_entry_digest": backend_digest,
                "execution": backend_entry["execution"],
            },
            "task_count": 2,
            "request_count": 16,
            "tasks": packet_tasks,
        },
        "packet_digest",
    )
    runtime_root = tmp_path / "runtime"
    runtime_request_path = tmp_path / "runtime-request.json"
    runtime_request = _sealed(
        runtime_request_path,
        {
            "schema_version": RUNTIME_REQUEST_SCHEMA_VERSION,
            "source_packet_digest": packet["packet_digest"],
            "backend": packet["backend"],
            "prompt_policy": "generic_masked_object_absent_background_completion_v2",
            "prompt": "Remove the selected object and reconstruct the empty room.",
            # This is the real production request shape: every frame and the
            # aggregate request count are bound, without a redundant per-task
            # camera_count field.
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "frames": deepcopy(task["frames"]),
                }
                for task in packet_tasks
            ],
            "retry_count": 0,
        },
        "request_digest",
    )
    runtime_tasks: list[dict[str, object]] = []
    teacher_records: list[dict[str, object]] = []
    for task in candidate["tasks"]:
        scene_root = Path(task["scene_directory"])
        frames: list[dict[str, object]] = []
        for frame in task["frames"]:
            source = scene_root / frame["rendered_rgb"]["relative_path"]
            output = runtime_root / "tasks" / task["task_id"] / f"{frame['frame_index']:05d}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.convert("RGB").save(output)
            record = _record(output, root=runtime_root)
            teacher_records.append(record)
            frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "camera_id": frame["camera_id"],
                    "semantic_teacher_frame": record,
                }
            )
        runtime_tasks.append(
            {
                "task_id": task["task_id"],
                "camera_count": len(frames),
                "frames": frames,
            }
        )
    runtime_result_path = runtime_root / f"{RUNTIME_RESULT_SCHEMA_VERSION}.json"
    runtime_result = _sealed(
        runtime_result_path,
        {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed_unreviewed_semantic_teacher_candidates",
            "source_runtime_request_digest": runtime_request["request_digest"],
            "backend_entry_digest": backend_digest,
            "task_count": 2,
            "request_count": 16,
            "tasks": runtime_tasks,
        },
        "result_digest",
    )
    import_path = tmp_path / "result-import.json"
    imported = _sealed(
        import_path,
        {
            "schema_version": "semantic_teacher_image_edit_result_import.v1",
            "status": "retained_unreviewed_semantic_teacher_candidates",
            "runtime_request": _record(runtime_request_path),
            "runtime_result": _record(runtime_result_path),
            "teacher_frames": teacher_records,
            "task_count": 2,
            "camera_count": 16,
            "all_generated_teacher_pngs_retained": True,
            "continuing_spend_from_this_run": False,
            "visual_reviewed": False,
            "appearance_qualified": False,
        },
        "result_import_digest",
    )
    return {
        "candidate_path": candidate_path,
        "packet_path": packet_path,
        "packet": packet,
        "runtime_root": runtime_root,
        "runtime_request_path": runtime_request_path,
        "runtime_request": runtime_request,
        "runtime_result_path": runtime_result_path,
        "runtime_result": runtime_result,
        "import_path": import_path,
        "imported": imported,
    }


def _run(fixture: dict[str, object], output: Path, *, radius: int = 2) -> dict:
    return materialize_semantic_teacher_artifixer_handoff(
        result_import_path=fixture["import_path"],
        semantic_teacher_packet_path=fixture["packet_path"],
        source_candidate_inputs_receipt_path=fixture["candidate_path"],
        transition_radius_pixels=radius,
        output_root=output,
    )


def _reseal_runtime_result(fixture: dict[str, object]) -> None:
    runtime_result = fixture["runtime_result"]
    _sealed(fixture["runtime_result_path"], runtime_result, "result_digest")
    imported = fixture["imported"]
    imported["runtime_result"] = _record(fixture["runtime_result_path"])
    _sealed(fixture["import_path"], imported, "result_import_digest")


def _reseal_runtime_request_and_dependents(fixture: dict[str, object]) -> None:
    runtime_request = fixture["runtime_request"]
    _sealed(fixture["runtime_request_path"], runtime_request, "request_digest")
    runtime_result = fixture["runtime_result"]
    runtime_result["source_runtime_request_digest"] = runtime_request["request_digest"]
    _sealed(fixture["runtime_result_path"], runtime_result, "result_digest")
    imported = fixture["imported"]
    imported["runtime_request"] = _record(fixture["runtime_request_path"])
    imported["runtime_result"] = _record(fixture["runtime_result_path"])
    _sealed(fixture["import_path"], imported, "result_import_digest")


def test_handoff_emits_exact_two_teacher_receipts_and_one_dual_packet(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff"

    result = _run(fixture, output)

    assert result["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert result["task_count"] == 2
    assert result["camera_count"] == 16
    assert len(result["semantic_teacher_receipts"]) == 2
    assert result["transition_radius_pixels"] == 2
    assert result["paid_execution_started"] is False
    assert result["appearance_qualified"] is False
    for row in result["semantic_teacher_receipts"]:
        receipt = json.loads((output / row["relative_path"]).read_text())
        assert receipt["schema_version"] == SEMANTIC_TEACHER_SCHEMA
        assert receipt["frame_count"] == 8
        assert receipt["appearance_repair_qualified"] is False
    dual = json.loads((output / result["dual_target_inputs"]["relative_path"]).read_text())
    assert dual["schema_version"] == SCHEMA_VERSION
    assert dual["replacement_object_count"] == 2
    assert dual["receipt_digest"] == result["dual_target_inputs"]["receipt_digest"]
    assert (output / f"{HANDOFF_SCHEMA_VERSION}.json").is_file()


def test_handoff_receipts_advance_fresh_scene_to_artifixer_execution(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "handoff"
    handoff = _run(fixture, output)
    candidate = json.loads(fixture["candidate_path"].read_text(encoding="utf-8"))
    task_ids = [task["task_id"] for task in candidate["tasks"]]
    task_freezes: list[Path] = []
    for slot, task_id in enumerate(task_ids, start=1):
        path = tmp_path / f"freeze-{task_id}.json"
        _write(path, _task(task_id, slot))
        task_freezes.append(path)
    artifacts: dict[str, object] = {}
    for contract in STAGE_CONTRACTS:
        if contract["stage_id"] == "artifixer3d_result":
            break
        if contract["stage_id"] == "semantic_teacher_receipts":
            artifacts[contract["stage_id"]] = {
                row["task_id"]: str(output / row["relative_path"])
                for row in handoff["semantic_teacher_receipts"]
            }
            continue
        if contract["stage_id"] == "dual_target_artifixer_inputs":
            artifacts[contract["stage_id"]] = str(
                output / handoff["dual_target_inputs"]["relative_path"]
            )
            continue
        value: dict[str, object] = {
            "schema_version": contract["schemas"][0],
            "status": (contract["accepted_statuses"][0] if contract["accepted_statuses"] else None),
        }
        if contract["digest_fields"]:
            field = contract["digest_fields"][0]
            value[field] = canonical_digest(value, digest_field=field)
        if contract["cardinality"] == "per_task":
            paths = {}
            for task_id in task_ids:
                path = tmp_path / f"{contract['stage_id']}-{task_id}.json"
                _write(path, value)
                paths[task_id] = str(path)
            artifacts[contract["stage_id"]] = paths
        else:
            path = tmp_path / f"{contract['stage_id']}.json"
            _write(path, value)
            artifacts[contract["stage_id"]] = str(path)

    status = materialize_fresh_scene_preparation_status(
        task_freeze_paths=task_freezes,
        artifacts=artifacts,
        output_path=tmp_path / "preparation-status.json",
    )

    assert status["first_blocker"] == "fresh_scene_artifixer3d_result_missing"
    assert status["next_required_stage"] == "artifixer3d_result"


def test_handoff_rejects_mutated_png_before_writing_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = next((fixture["runtime_root"] / "tasks").rglob("*.png"))
    frame.write_bytes(frame.read_bytes() + b"mutated")
    output = tmp_path / "must-not-write"

    with pytest.raises(DualTargetInputError, match="frame_inventory_invalid"):
        _run(fixture, output)

    assert not output.exists()


def test_handoff_rejects_runtime_camera_reorder_before_writing_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frames = fixture["runtime_result"]["tasks"][0]["frames"]
    frames[0]["camera_id"], frames[1]["camera_id"] = (
        frames[1]["camera_id"],
        frames[0]["camera_id"],
    )
    _reseal_runtime_result(fixture)
    output = tmp_path / "must-not-write"

    with pytest.raises(DualTargetInputError, match="task_camera_order_invalid"):
        _run(fixture, output)

    assert not output.exists()


def test_handoff_rejects_present_wrong_runtime_request_camera_count(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["runtime_request"]["tasks"][0]["camera_count"] = 7
    _reseal_runtime_request_and_dependents(fixture)
    output = tmp_path / "must-not-write"

    with pytest.raises(DualTargetInputError, match="task_camera_order_invalid"):
        _run(fixture, output)

    assert not output.exists()


def test_handoff_rejects_missing_or_duplicate_imported_frame(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    imported = fixture["imported"]
    imported["teacher_frames"][-1] = dict(imported["teacher_frames"][0])
    _sealed(fixture["import_path"], imported, "result_import_digest")
    output = tmp_path / "must-not-write"

    with pytest.raises(DualTargetInputError, match="frame_inventory_invalid"):
        _run(fixture, output)

    assert not output.exists()


def test_handoff_rejects_different_candidate_path_and_negative_radius(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    copied = tmp_path / "copied-candidate.json"
    shutil.copyfile(fixture["candidate_path"], copied)
    output = tmp_path / "must-not-write"

    with pytest.raises(DualTargetInputError, match="source_binding_invalid"):
        materialize_semantic_teacher_artifixer_handoff(
            result_import_path=fixture["import_path"],
            semantic_teacher_packet_path=fixture["packet_path"],
            source_candidate_inputs_receipt_path=copied,
            transition_radius_pixels=2,
            output_root=output,
        )
    with pytest.raises(DualTargetInputError, match="transition_radius_invalid"):
        _run(fixture, output, radius=-1)

    assert not output.exists()


def test_supervisor_binding_seals_handoff_request_and_all_transitive_inputs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    request_path = tmp_path / "handoff-request.json"
    request = materialize_fresh_scene_semantic_teacher_handoff_request(
        result_import_path=fixture["import_path"],
        semantic_teacher_packet_path=fixture["packet_path"],
        source_candidate_inputs_receipt_path=fixture["candidate_path"],
        transition_radius_pixels=2,
        output_path=request_path,
        roots=[tmp_path],
    )
    status_path = tmp_path / "status.json"
    status = _sealed(
        status_path,
        {
            "schema_version": "fresh_scene_paired_target_preparation.v1",
            "status": "blocked",
            "task_ids": ["task_0", "task_1"],
            "task_count": 2,
        },
        "status_digest",
    )
    manifest_path = tmp_path / "bindings.json"
    manifest = materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status_path,
        semantic_teacher_handoff_request_path=request_path,
        output_path=manifest_path,
        roots=[tmp_path],
    )

    assert request["transition_radius_pixels"] == 2
    assert (
        len(
            manifest["tool_requests"]["fresh_scene_semantic_teacher_handoff_request"][
                "input_inventory"
            ]
        )
        > 20
    )
    reopened = compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])
    assert (
        "materialize_fresh_scene_semantic_teacher_artifixer_handoff"
        in reopened["requested_tool_ids"]
    )
    assert (
        reopened["context_bindings"]["fresh_scene_semantic_teacher_handoff_request"][
            "request_digest"
        ]
        == request["request_digest"]
    )
    assert status["status_digest"]
