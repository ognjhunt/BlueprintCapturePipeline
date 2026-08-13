from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.fresh_scene_supervisor_bindings import (
    FreshSceneSupervisorBindingError,
    compile_fresh_scene_supervisor_bindings,
    materialize_fresh_scene_supervisor_bindings,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _status(path: Path) -> Path:
    value = {
        "schema_version": "fresh_scene_paired_target_preparation.v1",
        "status": "blocked",
        "task_count": 1,
        "task_ids": ["task_a"],
        "first_blocker": "fresh_scene_sam31_task_inputs_missing",
        "next_required_stage": "sam31_task_inputs",
        "status_digest": "",
    }
    value["status_digest"] = canonical_digest(value, digest_field="status_digest")
    return _write(path, value)


def _sam_request(root: Path) -> Path:
    for name in ("calibrated.json", "task.json", "profile.json", "prompts.json"):
        _write(root / name, {})
    ffmpeg = root / "ffmpeg"
    ffmpeg.write_bytes(b"fixture")
    value = {
        "schema_version": "fresh_scene_sam31_task_input_tool_request.v1",
        "calibrated_view_receipt_path": str(root / "calibrated.json"),
        "task_freeze_path": str(root / "task.json"),
        "provider_profile_path": str(root / "profile.json"),
        "prompts_path": str(root / "prompts.json"),
        "ffmpeg_executable": str(ffmpeg),
        "frame_rate_hz": 1,
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return _write(root / "sam-request.json", value)


def test_host_resident_manifest_compiles_exact_agents_sdk_bindings(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _sam_request(tmp_path / "inputs")
    manifest_path = tmp_path / "binding.json"

    manifest = materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        sam31_task_input_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    compiled = compile_fresh_scene_supervisor_bindings(
        manifest_path, roots=[tmp_path]
    )

    assert manifest["agent_receives_paths"] is False
    assert manifest["paid_execution_authorized"] is False
    assert manifest["provider_mutations_performed"] == 0
    assert compiled["requested_tool_ids"] == [
        "inspect_fresh_scene_preparation",
        "materialize_sam31_task_inputs",
    ]
    assert set(compiled["context_bindings"]) == {
        "fresh_scene_preparation_status",
        "fresh_scene_sam31_task_input_request",
    }
    assert (
        compiled["context_bindings"]["fresh_scene_sam31_task_input_request"][
            "request_digest"
        ]
        == json.loads(request.read_text())["request_digest"]
    )


def test_binding_rejects_outside_root_and_changed_request(tmp_path: Path) -> None:
    resident = tmp_path / "resident"
    resident.mkdir()
    status = _status(resident / "status.json")
    outside = _sam_request(tmp_path / "outside")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_not_host_resident",
    ):
        materialize_fresh_scene_supervisor_bindings(
            preparation_status_path=status,
            sam31_task_input_request_path=outside,
            output_path=resident / "bad-binding.json",
            roots=[resident],
        )

    request = _sam_request(resident / "inputs")
    manifest_path = resident / "binding.json"
    materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        sam31_task_input_request_path=request,
        output_path=manifest_path,
        roots=[resident],
    )
    request.write_text("{}", encoding="utf-8")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[resident])


def test_binding_rejects_changed_request_input_bytes(tmp_path: Path) -> None:
    status = _status(tmp_path / "status.json")
    request = _sam_request(tmp_path / "inputs")
    manifest_path = tmp_path / "binding.json"
    materialize_fresh_scene_supervisor_bindings(
        preparation_status_path=status,
        sam31_task_input_request_path=request,
        output_path=manifest_path,
        roots=[tmp_path],
    )
    (tmp_path / "inputs/prompts.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(
        FreshSceneSupervisorBindingError,
        match="fresh_scene_tool_request_input_bytes_changed",
    ):
        compile_fresh_scene_supervisor_bindings(manifest_path, roots=[tmp_path])
