from __future__ import annotations

from pathlib import Path
import json

from blueprint_pipeline.capture_orchestrator import (
    PipelineConfig,
    _build_derived_lane_result,
    resolve_requested_lanes,
    run_capture_pipeline,
)


def test_build_derived_lane_result_preserves_current_orchestration_shape() -> None:
    result = _build_derived_lane_result(
        lane="evaluation_prep",
        source="evaluation_prep_artifacts",
        qualification_result={
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
        extra_fields={"manifest_path": "pipeline/evaluation_prep/evaluation_prep_manifest.json"},
    )

    assert result == {
        "status": "completed",
        "lane": "evaluation_prep",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        "source": "evaluation_prep_artifacts",
        "manifest_path": "pipeline/evaluation_prep/evaluation_prep_manifest.json",
    }


def test_capture_orchestrator_keeps_supported_lanes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["qualification", "scene_memory", "evaluation_prep"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json",
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )
    assert result["lanes"] == ["qualification", "scene_memory", "evaluation_prep"]
    assert all(item["lane"] != "advanced_geometry" for item in result["results"])


def test_capture_orchestrator_current_lane_runs_simulation_automation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        lane="current",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert result["lanes"] == ["qualification", "evaluation_prep", "simulation_automation"]
    assert [item["lane"] for item in result["results"]] == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]
    assert result["results"][-1]["automation_status"] == "blocked"
    assert result["results"][-1]["robot_eval_job_inbox_status"] == "waiting_for_job_requests"
    assert result["results"][-1]["robot_eval_job_inbox_processed_count"] == 0


def test_capture_orchestrator_processes_robot_eval_job_inbox_when_present(
    monkeypatch,
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")
    inbox = descriptor_path.parent / "pipeline" / "robot_eval_job_requests" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "robot-eval-job.json").write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_evaluation_prep_stage",
        lambda **_kwargs: {"manifest_path": str(tmp_path / "evaluation_prep_manifest.json")},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.build_simulation_automation",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "simulation_automation_run_manifest.json"),
            "plan_path": str(tmp_path / "simulation_automation_plan.json"),
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    def _run_inbox(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return {"status": "completed", "processed_count": 1}

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.run_robot_eval_job_request_inbox",
        _run_inbox,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        lane="current",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert len(calls) == 1
    assert calls[0]["capture_root"] == descriptor_path.parent
    assert calls[0]["inbox_dir"] == inbox
    assert result["results"][-1]["robot_eval_job_inbox_status"] == "completed"
    assert result["results"][-1]["robot_eval_job_inbox_processed_count"] == 1


def test_capture_orchestrator_runs_single_capture_smoke_lane(monkeypatch, tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_requested_lanes",
        lambda **_kwargs: ["cosmos_single_capture_smoke"],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_benchmark.run_cosmos_single_capture_smoke_lane",
        lambda **_kwargs: {"status": "blocked", "reason": "runtime_unavailable"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.capture_orchestrator.resolve_gs_uri_to_path",
        lambda *_args, **_kwargs: descriptor_path,
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    assert result["lanes"] == ["cosmos_single_capture_smoke"]
    assert result["results"] == [
        {
            "lane": "cosmos_single_capture_smoke",
            "status": "blocked",
            "reason": "runtime_unavailable",
        }
    ]


def test_resolve_requested_lanes_defaults_to_current_stack_for_site_world_candidate(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
                "requested_outputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification", "evaluation_prep", "simulation_automation"]


def test_resolve_requested_lanes_honors_explicit_descriptor_requested_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": [
                    "qualification",
                    "retrieval_index",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": [],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_accepts_capture_bridge_robot_eval_alias_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
                "requested_lanes": [
                    "evaluation_prep",
                    "robot_eval_dataset",
                    "task_evaluation_run",
                ],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification", "evaluation_prep", "simulation_automation"]


def test_resolve_requested_lanes_infers_current_lanes_from_robot_eval_outputs(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification", "evaluation_prep", "simulation_automation"]


def test_resolve_requested_lanes_demotes_bridge_default_scene_memory_pair(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": ["qualification", "scene_memory"],
                "requested_outputs": [],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == ["qualification"]


def test_resolve_requested_lanes_prefers_explicit_descriptor_lanes_over_output_inference(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requested_lanes": [
                    "qualification",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": ["preview_simulation", "deeper_evaluation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_prefers_explicit_descriptor_lanes_over_native_candidate_default(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
                "requested_lanes": [
                    "qualification",
                    "synthesis_coverage_validation",
                ],
                "requested_outputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "synthesis_coverage_validation",
    ]


def test_resolve_requested_lanes_accepts_camel_case_descriptor_fields(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requestedLanes": [
                    "qualification",
                    "retrieval_index",
                ],
                "requestedOutputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
    ]


def test_resolve_requested_lanes_accepts_scalar_descriptor_requested_lanes(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "requestedLanes": "retrieval_index",
                "requestedOutputs": ["preview_simulation"],
            }
        ),
        encoding="utf-8",
    )

    lanes = resolve_requested_lanes(
        descriptor_gcs_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        gcs_root=tmp_path,
    )

    assert lanes == [
        "qualification",
        "retrieval_index",
    ]
