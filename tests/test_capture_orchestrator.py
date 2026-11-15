from __future__ import annotations

import json

import pytest

from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline

_DESCRIPTOR_URI = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"


def _capture(tmp_path, descriptor=None):
    root = tmp_path / "bucket/scenes/scene-1/captures/capture-1"
    root.mkdir(parents=True)
    (root / "capture_descriptor.json").write_text(json.dumps(descriptor or {}))
    return root


@pytest.mark.parametrize("descriptor", [
    {},
    {"requested_outputs": ["task_evaluation_run"]},
    {"requested_lanes": ["qualification", "evaluation_prep", "simulation_automation"]},
    {"capture_mode": {"resolved_mode": "site_world_candidate"},
     "scene_memory_capture": {"world_model_candidate": True}},
])
def test_every_capture_runs_only_the_qualification_lane(monkeypatch, tmp_path, descriptor):
    root = _capture(tmp_path, descriptor)
    monkeypatch.setenv("BLUEPRINT_SIM_ONLY_BETA_AUTONOMY", "1")
    calls = []
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
                        lambda **kw: calls.append(kw) or {"status": "completed", "lane": "qualification"})

    result = run_capture_pipeline(descriptor_gcs_uri=_DESCRIPTOR_URI, config=PipelineConfig(gcs_root=tmp_path))

    assert len(calls) == 1 and calls[0]["requested_lanes"] == ["qualification"]
    assert result == {
        "status": "completed",
        "artifact_purpose": "evaluation_preparation",
        "robot_evaluation_performed": False,
        "descriptor_gcs_uri": _DESCRIPTOR_URI,
        "bucket": "bucket",
        "lanes": ["qualification"],
        "results": [{"status": "completed", "lane": "qualification"}],
    }
    assert not (root / "pipeline/robot_eval_job_requests").exists()


@pytest.mark.parametrize("request_kwargs", [
    {"lane": "evaluation_prep"},
    {"lane": "all"},
    {"requested_lanes": ["qualification", "retrieval_index"]},
    {"requested_lanes": "cosmos_single_capture_smoke"},
])
def test_any_other_lane_is_refused_before_work(monkeypatch, tmp_path, request_kwargs):
    _capture(tmp_path)
    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
                        lambda **_kw: pytest.fail("qualification must not start"))
    with pytest.raises(ValueError, match="Unsupported pipeline lane"):
        run_capture_pipeline(descriptor_gcs_uri=_DESCRIPTOR_URI, config=PipelineConfig(gcs_root=tmp_path),
                             **request_kwargs)


def test_qualification_failure_propagates(monkeypatch, tmp_path):
    _capture(tmp_path)

    def failed(**_kwargs):
        raise RuntimeError("qualification failed")

    monkeypatch.setattr("blueprint_pipeline.capture_orchestrator.run_qualification_pipeline", failed)
    with pytest.raises(RuntimeError, match="qualification failed"):
        run_capture_pipeline(descriptor_gcs_uri=_DESCRIPTOR_URI, lane="qualification",
                             config=PipelineConfig(gcs_root=tmp_path))
