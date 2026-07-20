from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.lane_resume import (
    LANE_LEDGER_SCHEMA_VERSION,
    capture_input_fingerprint,
    lane_marker_path,
    lane_resume_disabled,
    read_completed_lane_result,
    record_lane_completion,
)


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1"
    capture_root.mkdir(parents=True, exist_ok=True)
    return capture_root


def _write_capture_inputs(capture_root: Path, *, descriptor_body: str = "{}") -> Path:
    descriptor_path = capture_root / "capture_descriptor.json"
    descriptor_path.write_text(descriptor_body, encoding="utf-8")
    raw_root = capture_root / "raw"
    raw_root.mkdir(exist_ok=True)
    (raw_root / "manifest.json").write_text('{"files": []}', encoding="utf-8")
    return descriptor_path


def _fingerprint(capture_root: Path) -> dict:
    return capture_input_fingerprint(
        capture_root=capture_root,
        descriptor_path=capture_root / "capture_descriptor.json",
    )


def test_capture_input_fingerprint_is_deterministic_and_input_sensitive(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    _write_capture_inputs(capture_root)

    first = _fingerprint(capture_root)
    second = _fingerprint(capture_root)
    assert first == second
    assert first["schema_version"] == "run_e2e_capture_input_fingerprint.v1"
    assert first["fingerprint_sha256"]

    (capture_root / "raw" / "manifest.json").write_text(
        '{"files": ["frame_0001.jpg"]}', encoding="utf-8"
    )
    changed = _fingerprint(capture_root)
    assert changed["fingerprint_sha256"] != first["fingerprint_sha256"]


def test_lane_marker_round_trip_returns_stored_result(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    _write_capture_inputs(capture_root)
    fingerprint = _fingerprint(capture_root)
    manifest_path = capture_root / "pipeline" / "evaluation_prep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    lane_result = {
        "lane": "evaluation_prep",
        "status": "completed",
        "manifest_path": str(manifest_path),
    }

    record_lane_completion(
        capture_root=capture_root,
        lane="evaluation_prep",
        fingerprint=fingerprint,
        lane_result=lane_result,
    )

    marker_path = lane_marker_path(capture_root, "evaluation_prep")
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["schema_version"] == LANE_LEDGER_SCHEMA_VERSION
    assert marker["output_paths"] == [str(manifest_path)]

    resumed = read_completed_lane_result(
        capture_root=capture_root,
        lane="evaluation_prep",
        fingerprint=fingerprint,
    )
    assert resumed == lane_result


def test_lane_marker_fingerprint_mismatch_forces_rerun(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    _write_capture_inputs(capture_root)
    fingerprint = _fingerprint(capture_root)
    record_lane_completion(
        capture_root=capture_root,
        lane="qualification",
        fingerprint=fingerprint,
        lane_result={"lane": "qualification", "status": "completed"},
    )

    _write_capture_inputs(capture_root, descriptor_body='{"changed": true}')
    changed_fingerprint = _fingerprint(capture_root)

    assert (
        read_completed_lane_result(
            capture_root=capture_root,
            lane="qualification",
            fingerprint=changed_fingerprint,
        )
        is None
    )


def test_lane_marker_missing_outputs_force_rerun(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    _write_capture_inputs(capture_root)
    fingerprint = _fingerprint(capture_root)
    manifest_path = capture_root / "pipeline" / "simulation_automation_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    record_lane_completion(
        capture_root=capture_root,
        lane="simulation_automation",
        fingerprint=fingerprint,
        lane_result={
            "lane": "simulation_automation",
            "status": "completed",
            "manifest_path": str(manifest_path),
        },
    )

    manifest_path.unlink()

    assert (
        read_completed_lane_result(
            capture_root=capture_root,
            lane="simulation_automation",
            fingerprint=fingerprint,
        )
        is None
    )


def test_lane_resume_kill_switch(monkeypatch) -> None:
    monkeypatch.delenv("BLUEPRINT_LANE_RESUME_DISABLED", raising=False)
    assert lane_resume_disabled() is False
    monkeypatch.setenv("BLUEPRINT_LANE_RESUME_DISABLED", "1")
    assert lane_resume_disabled() is True
    monkeypatch.setenv("BLUEPRINT_LANE_RESUME_DISABLED", "false")
    assert lane_resume_disabled() is False


def _orchestrator_lane_stubs(monkeypatch, descriptor_path: Path, calls: dict[str, int]):
    def _qualification(**_kwargs):
        calls["qualification"] = calls.get("qualification", 0) + 1
        return {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        }

    def _evaluation_prep(**_kwargs):
        calls["evaluation_prep"] = calls.get("evaluation_prep", 0) + 1
        manifest_path = descriptor_path.parent / "pipeline" / "evaluation_prep_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        return {"manifest_path": str(manifest_path)}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["qualification", "evaluation_prep"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        _qualification,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        _evaluation_prep,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )


def test_orchestrator_retry_skips_completed_lanes(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    descriptor_path = _write_capture_inputs(capture_root)
    calls: dict[str, int] = {}
    _orchestrator_lane_stubs(monkeypatch, descriptor_path, calls)

    descriptor_gcs_uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    first = run_capture_pipeline(
        descriptor_gcs_uri=descriptor_gcs_uri,
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert first["status"] == "completed"
    assert calls == {"qualification": 1, "evaluation_prep": 1}

    retry = run_capture_pipeline(
        descriptor_gcs_uri=descriptor_gcs_uri,
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert retry["status"] == "completed"
    # Retry with an unchanged capture input must not redo completed lanes.
    assert calls == {"qualification": 1, "evaluation_prep": 1}
    assert [item["lane"] for item in retry["results"]] == ["qualification", "evaluation_prep"]
    assert all(item.get("resumed_from_lane_ledger") is True for item in retry["results"])


def test_orchestrator_kill_switch_bypasses_lane_ledger(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    descriptor_path = _write_capture_inputs(capture_root)
    calls: dict[str, int] = {}
    _orchestrator_lane_stubs(monkeypatch, descriptor_path, calls)
    monkeypatch.setenv("BLUEPRINT_LANE_RESUME_DISABLED", "1")

    descriptor_gcs_uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_gcs_uri,
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert not lane_marker_path(capture_root, "qualification").exists()
    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_gcs_uri,
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert calls == {"qualification": 2, "evaluation_prep": 2}


def test_orchestrator_changed_input_reruns_lanes(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    descriptor_path = _write_capture_inputs(capture_root)
    calls: dict[str, int] = {}
    _orchestrator_lane_stubs(monkeypatch, descriptor_path, calls)

    descriptor_gcs_uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_gcs_uri,
        config=PipelineConfig(gcs_root=tmp_path),
    )
    _write_capture_inputs(capture_root, descriptor_body='{"recaptured": true}')
    run_capture_pipeline(
        descriptor_gcs_uri=descriptor_gcs_uri,
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert calls == {"qualification": 2, "evaluation_prep": 2}
