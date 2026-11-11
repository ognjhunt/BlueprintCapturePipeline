from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import blueprint_pipeline.capture_orchestrator as co


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _descriptor_uri(root: Path, payload: dict | None = None) -> str:
    uri = "gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json"
    descriptor = {
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "raw_prefix_uri": "gs://bucket/scenes/scene-1/captures/capture-1/raw",
        "requested_outputs": ["qualification"],
    }
    if payload:
        descriptor.update(payload)
    _write_json(root / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json", descriptor)
    return uri


def test_lane_descriptor_and_runtime_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert co._normalize_lane_value(" ") is None
    with pytest.raises(ValueError):
        co._normalize_lane_value("unsupported")
    with pytest.raises(ValueError):
        co._normalize_requested_lanes(123)
    assert co._normalize_requested_lanes([" ", "simulation_automation"]) == [
        "qualification",
        "evaluation_prep",
        "simulation_automation",
    ]
    assert co._mapping_value({"capture_id": "direct"}, "capture_id") == "direct"
    assert co._mapping_value({"metadata": {"capture_id": "from-meta"}}, "capture_id") == "from-meta"
    assert co._mapping_value({"capture_bundle": {"capture_id": "from-bundle"}}, "capture_id") == "from-bundle"
    assert co._descriptor_requested_outputs({"requested_outputs": "task_evaluation_run"}) == {
        "task_evaluation_run"
    }
    assert co._descriptor_is_android_xr_video_only(
        {"metadata": {"capture_profile_id": "android_xr_glasses"}}
    )
    assert not co._descriptor_is_native_default_candidate(
        {"metadata": {"capture_profile_id": "android_xr_glasses"}}
    )
    assert co._descriptor_is_native_default_candidate(
        {
            "metadata": {
                "capture_mode": {"resolved_mode": "site_world_candidate"},
                "scene_memory_capture": {"world_model_candidate": True},
            }
        }
    )

    root = tmp_path / "gcs"
    android_uri = _descriptor_uri(
        root,
        {"metadata": {"capture_modality": "android_xr_video_only"}},
    )
    assert co._load_descriptor_requested_lanes(android_uri, root) == ["qualification"]
    robot_eval_uri = _descriptor_uri(root, {"requested_outputs": ["robot_eval_dataset"]})
    assert co._load_descriptor_requested_lanes(robot_eval_uri, root) == [
        "qualification",
        "evaluation_prep",
    ]
    preview_uri = _descriptor_uri(root, {"requested_outputs": ["preview_simulation"]})
    assert co._load_descriptor_requested_lanes(preview_uri, root) == list(co._CURRENT_PIPELINE_LANES)
    scene_memory_uri = _descriptor_uri(root, {"requested_outputs": ["scene_memory"]})
    assert co._load_descriptor_requested_lanes(scene_memory_uri, root) == [
        "qualification",
        "scene_memory",
    ]
    monkeypatch.setenv(co.SIM_ONLY_BETA_AUTONOMY_ENV, "true")
    assert co._load_descriptor_requested_lanes(_descriptor_uri(root, {"requested_outputs": []}), root) == list(
        co._CURRENT_PIPELINE_LANES
    )
    monkeypatch.setenv("PIPELINE_LANE", "evaluation_prep")
    assert co.resolve_requested_lanes(descriptor_gcs_uri=android_uri, gcs_root=root) == [
        "qualification",
        "evaluation_prep",
    ]

    monkeypatch.delenv(co.SIM_ONLY_BETA_AUTONOMY_ENV, raising=False)
    assert co._load_descriptor_requested_lanes(_descriptor_uri(root, {"requested_outputs": []}), root) == [
        "qualification"
    ]


def test_capture_pipeline_dispatch_and_synthesis_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "gcs"
    descriptor_uri = _descriptor_uri(
        root,
        {"quality": {"world_model_candidate": True}, "metadata": {"site_identity": {"site_id": "site-1"}}},
    )
    cfg = co.PipelineConfig(gcs_root=root)
    original_synthesis_wrapper = co._run_synthesis_coverage_validation
    original_synthesis_validation = co.run_capture_synthesis_validation
    monkeypatch.setattr(
        co,
        "run_qualification_pipeline",
        lambda **_kwargs: {
            "status": "completed",
            "lane": "qualification",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "pipeline_prefix": "scenes/scene-1/captures/capture-1/pipeline",
        },
    )
    monkeypatch.setattr(co, "run_retrieval_index_stage", lambda **_kwargs: {"status": "retrieved"})
    monkeypatch.setattr(co, "run_frame_alignment_stage", lambda **_kwargs: {"status": "aligned"})
    monkeypatch.setattr(co, "_run_synthesis_coverage_validation", lambda **_kwargs: {"status": "validated"})
    cosmos_module = ModuleType("blueprint_pipeline.synthesis.cosmos_benchmark")
    cosmos_module.run_cosmos_single_capture_smoke_lane = lambda **_kwargs: {"status": "smoked"}
    monkeypatch.setitem(sys.modules, "blueprint_pipeline.synthesis.cosmos_benchmark", cosmos_module)
    result = co.run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        requested_lanes=[
            "retrieval_index",
            "frame_alignment",
            "synthesis_coverage_validation",
            "cosmos_single_capture_smoke",
        ],
        allow_legacy_lanes=True,
        config=cfg,
    )
    assert result["lanes"] == [
        "qualification",
        "retrieval_index",
        "frame_alignment",
        "synthesis_coverage_validation",
        "cosmos_single_capture_smoke",
    ]
    assert [item["lane"] for item in result["results"]][-1] == "cosmos_single_capture_smoke"
    monkeypatch.setattr(co, "resolve_requested_lanes", lambda **_kwargs: ["unsupported"])
    with pytest.raises(ValueError):
        co.run_capture_pipeline(descriptor_gcs_uri=descriptor_uri, config=cfg)
    monkeypatch.setattr(co, "resolve_requested_lanes", lambda **_kwargs: ["evaluation_prep"])
    monkeypatch.setattr(co, "run_evaluation_prep_stage", lambda **_kwargs: {"manifest_path": "prep.json"})
    eval_only = co.run_capture_pipeline(descriptor_gcs_uri=descriptor_uri, config=cfg)
    assert eval_only["results"][0]["lane"] == "evaluation_prep"
    monkeypatch.setattr(co, "_run_synthesis_coverage_validation", original_synthesis_wrapper)
    monkeypatch.setattr(co, "run_capture_synthesis_validation", lambda **_kwargs: {"status": "validated"})
    assert co._run_synthesis_coverage_validation(
        capture_root=co.resolve_gs_uri_to_path(descriptor_uri, root).parent,
        descriptor_gcs_uri=descriptor_uri,
        cfg=cfg,
    ) == {"status": "validated"}
    monkeypatch.setattr(co, "run_capture_synthesis_validation", original_synthesis_validation)

    unreadable_uri = _descriptor_uri(root)
    descriptor_path = co.resolve_gs_uri_to_path(unreadable_uri, root)
    descriptor_path.write_text("{bad", encoding="utf-8")
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["status"] == "failed"
    descriptor_path.write_text(json.dumps({"quality": {"world_model_candidate": False}}), encoding="utf-8")
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["reason"] == "not_world_model_candidate"
    descriptor_path.write_text(json.dumps({"world_model_candidate": True}), encoding="utf-8")
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["reason"] == "no_site_id_in_descriptor"

    descriptor_path.write_text(
        json.dumps(
            {
                "world_model_candidate": True,
                "site_id": "site-1",
                "capture_id": "capture-1",
                "metadata": {"capture_topology": {"pass_id": "pass-1"}},
            }
        ),
        encoding="utf-8",
    )
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["reason"] == "no_site_reference_index"
    index = root / "bucket" / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text("{bad\n", encoding="utf-8")
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["status"] == "failed"
    index.write_text(json.dumps({"pass_id": "pass-1"}) + "\n", encoding="utf-8")
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["reason"] == "no_prior_pass_in_index"
    index.write_text(json.dumps({"pass_id": "pass-0", "site_frame_transform": None}) + "\n", encoding="utf-8")
    monkeypatch.setattr(co, "load_capture_geometry", lambda **_kwargs: {"poses": []})
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["reason"] == "no_geometry_poses"
    monkeypatch.setattr(co, "load_capture_geometry", lambda **_kwargs: {"poses": [{"transform": [1, 2, 3]}]})
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["reason"] == "invalid_pose_shape"
    monkeypatch.setattr(
        co,
        "load_capture_geometry",
        lambda **_kwargs: {"poses": [{"transform": list(range(16))}], "intrinsics": None},
    )
    monkeypatch.setattr(co, "synthesize_view", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
    )["status"] == "failed"
    monkeypatch.setattr(
        co,
        "synthesize_view",
        lambda **_kwargs: {"status": "completed", "coverage_frac": 0.7, "retrieval_dist_m": 1.2},
    )
    completed = co.run_capture_synthesis_validation(
        capture_root=descriptor_path.parent,
        descriptor_gcs_uri=unreadable_uri,
        cfg=cfg,
        mode="cosmos_i2w",
    )
    assert completed["status"] == "completed"
    assert completed["output_video_uri"].endswith("capture-1_cosmos.mp4")
