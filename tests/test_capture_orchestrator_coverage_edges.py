from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from blueprint_pipeline.common import PipelineError
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

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "job.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_INBOX_DIR", str(inbox))
    assert co._robot_eval_job_request_inbox_for_capture(tmp_path) == inbox

    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_INBOX_DIR", raising=False)
    monkeypatch.delenv(co.SIM_ONLY_BETA_AUTONOMY_ENV, raising=False)
    assert co._load_descriptor_requested_lanes(_descriptor_uri(root, {"requested_outputs": []}), root) == [
        "qualification"
    ]
    assert co._env_int("MISSING_INT", 7) == 7
    monkeypatch.setenv("BAD_INT", "not-int")
    assert co._env_int("BAD_INT", 7) == 7
    monkeypatch.setenv("NEGATIVE_INT", "-1")
    assert co._env_int("NEGATIVE_INT", 7) == 7
    monkeypatch.setenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR_COMMAND", "custom sim")
    assert co._mujoco_beta_simulator_command(tmp_path) == "custom sim"
    monkeypatch.delenv("ROBOT_EVAL_JOB_DEFAULT_SIMULATOR_COMMAND", raising=False)
    monkeypatch.setenv(co.MUJOCO_G1_MODEL_ROOT_ENV, "/tmp/g1 root")
    monkeypatch.setenv(co.MUJOCO_BETA_SKIP_RENDER_ENV, "true")
    assert "--skip-render-frames" in co._mujoco_beta_simulator_command(tmp_path)
    monkeypatch.delenv(co.MUJOCO_G1_MODEL_ROOT_ENV, raising=False)
    monkeypatch.setenv(co.MUJOCO_ALLOW_FETCH_G1_ASSETS_ENV, "true")
    assert "--allow-fetch-g1-assets" in co._mujoco_beta_simulator_command(tmp_path)
    monkeypatch.delenv(co.MUJOCO_ALLOW_FETCH_G1_ASSETS_ENV, raising=False)
    assert co._mujoco_beta_simulator_command(tmp_path) == ""
    monkeypatch.setenv("ROBOT_EVAL_JOB_ALLOWED_SIMULATORS", "mujoco;isaac")
    monkeypatch.setenv(co.ALLOW_SIMULATOR_EXECUTION_ENV, "true")
    runtime = co._default_robot_eval_job_runtime(tmp_path)
    assert runtime["allowed_simulators"] == ["mujoco", "isaac"]


def test_robot_eval_artifact_parsers_and_validators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert co._load_cards(tmp_path / "missing.json") == []
    cards_path = tmp_path / "cards.json"
    _write_json(cards_path, {"cards": [{"id": "a"}, "bad"]})
    assert co._load_cards(cards_path) == [{"id": "a"}]

    metrics = tmp_path / "metrics.json"
    _write_json(metrics, {"metrics": {"a": {"metricId": "metric-a"}, "metric-b": "scalar"}})
    assert co._metric_ids_from_methodology(metrics) == {"metric-a", "metric-b"}
    _write_json(metrics, {"metrics": [{"id": "metric-c"}, "metric-d"]})
    assert co._metric_ids_from_methodology(metrics) == {"metric-c", "metric-d"}
    _write_json(metrics, {"metrics": None})
    assert co._metric_ids_from_methodology(metrics) == set()
    thresholds = tmp_path / "thresholds.json"
    _write_json(thresholds, {"thresholds": {"a": {"taskId": "task-a"}, "task-b": 1}})
    assert co._task_ids_from_thresholds(thresholds) == {"task-a", "task-b"}
    _write_json(thresholds, {"thresholds": ["bad", {"id": "task-c"}]})
    assert co._task_ids_from_thresholds(thresholds) == {"task-c"}
    _write_json(thresholds, {"thresholds": None})
    assert co._task_ids_from_thresholds(thresholds) == set()
    taxonomy = tmp_path / "taxonomy.json"
    assert co._failure_mode_ids_from_taxonomy(taxonomy) == set()
    _write_json(taxonomy, {"failureModes": [{"id": "drop"}, "timeout"]})
    assert co._failure_mode_ids_from_taxonomy(taxonomy) == {"drop", "timeout"}

    outside = tmp_path.parent / "outside.txt"
    assert co._relative_to(tmp_path, outside) == str(outside)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("data", encoding="utf-8")
    assert co._local_artifact(artifact, base_dir=tmp_path)["size_bytes"] == 4
    capture_root = tmp_path / "capture"
    automation_root = capture_root / "pipeline" / "simulation_automation"
    automation_root.mkdir(parents=True)
    local = automation_root / "local.json"
    local.write_text("{}", encoding="utf-8")
    assert co._automation_local_reference_path(f"file://{local}", capture_root=capture_root, automation_root=automation_root) == local
    assert co._automation_local_reference_path(str(local), capture_root=capture_root, automation_root=automation_root) == local
    assert co._automation_local_reference_path("https://example/input.json", capture_root=capture_root, automation_root=automation_root) is None
    assert co._automation_local_reference_path("local.json", capture_root=capture_root, automation_root=automation_root) == local
    assert co._automation_local_reference_path("pipeline/file.json", capture_root=capture_root, automation_root=automation_root) == capture_root / "pipeline" / "file.json"
    monkeypatch.setenv("GCS_ROOT", str(tmp_path / "gcs"))
    assert co._automation_local_reference_path(
        "gs://bucket/path/input.json",
        capture_root=capture_root,
        automation_root=automation_root,
    ) == tmp_path / "gcs" / "path" / "input.json"

    assert co._string_list("value") == ["value"]
    assert co._string_list(123) == []
    family_path = tmp_path / "families.json"
    assert co._scenario_family_rows(family_path) == []
    _write_json(family_path, {"families": [{"family_id": "f1"}, "bad"]})
    assert co._scenario_family_rows(family_path) == [{"family_id": "f1"}]
    assert co._scenario_family_variation_names({}) == set()
    assert co._scenario_family_variation_names({"variations": ["bad", {"variationId": "v1"}]}) == {"v1"}
    instances_path = tmp_path / "instances.json"
    assert co._variation_instance_rows(instances_path) == []
    _write_json(instances_path, {"instances": [{"scenario_id": "s1"}, "bad"]})
    assert co._variation_instance_rows(instances_path) == [{"scenario_id": "s1"}]

    capture_root = tmp_path / "capture"
    validation = co._validate_scenario_variation_artifacts(
        capture_root=capture_root,
        requested_tasks=[{"task_id": "task-1", "scenario_ids": ["scenario-1"]}],
    )
    assert "robot_eval_scenario_family_library_missing" in validation["blockers"]
    assert "robot_eval_scenario_variation_instances_missing" in validation["blockers"]
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    automation_root = capture_root / "pipeline" / "simulation_automation"
    _write_json(
        robot_eval_root / "scenario_family_library.json",
        {"families": []},
    )
    _write_json(
        automation_root / "scenario_variation_instances.json",
        {"status": "completed", "instances": []},
    )
    empty_validation = co._validate_scenario_variation_artifacts(
        capture_root=capture_root,
        requested_tasks=[{"task_id": "task-1", "scenario_ids": ["scenario-1"]}],
    )
    assert "robot_eval_scenario_family_library_empty" in empty_validation["blockers"]
    assert "robot_eval_scenario_variation_instances_empty" in empty_validation["blockers"]
    _write_json(
        robot_eval_root / "scenario_family_library.json",
        {
            "families": [
                {
                    "family_id": "fam-1",
                    "task_id": "different",
                    "scenario_id": "scenario-x",
                    "variations": [{"variation_name": "baseline"}],
                }
            ]
        },
    )
    _write_json(
        automation_root / "scenario_variation_instances.json",
        {"status": "failed", "instances": [{"scenario_id": "scenario-1", "variation_id": "baseline"}]},
    )
    validation = co._validate_scenario_variation_artifacts(
        capture_root=capture_root,
        requested_tasks=[{"task_id": "task-1", "scenario_ids": ["scenario-1"]}],
    )
    assert "robot_eval_scenario_family_library_missing_task_coverage" in validation["blockers"]
    assert "robot_eval_scenario_variation_instances_not_completed" in validation["blockers"]

    dataset_validation = co._validate_robot_eval_dataset_required_inputs(capture_root=capture_root)
    assert dataset_validation["missing_robot_eval_dataset_inputs"]

    registry_missing = co._validate_simulator_engine_plugin_registry(capture_root=capture_root)
    assert "robot_eval_simulator_engine_plugin_registry_missing" in registry_missing["blockers"]
    _write_json(automation_root / "simulator_engine_plugin_registry.json", {"plugins": {}, "world_model_plugins": {}})
    registry_empty = co._validate_simulator_engine_plugin_registry(capture_root=capture_root)
    assert "robot_eval_simulator_engine_plugin_registry_empty" in registry_empty["blockers"]
    all_sim_inputs = {
        key: "missing-local.json"
        for key in co._required_simulator_plugin_input_keys("isaac_lab_arena")
    }
    all_world_inputs = {
        key: "missing-world.json"
        for key in co._required_world_model_plugin_input_keys("site_world_runtime")
    }
    _write_json(
        automation_root / "simulator_engine_plugin_registry.json",
        {
            "plugins": {
                "isaac_lab_arena": {
                    "adapter_contract_status": "draft",
                    "managed_execution_supported": False,
                    "inputs": all_sim_inputs,
                }
            },
            "world_model_plugins": {
                "worldlabs_world_model": {
                    "adapter_contract_status": "draft",
                    "managed_execution_supported": False,
                    "source_status": "ready",
                    "inputs": all_world_inputs,
                }
            },
        },
    )
    registry = co._validate_simulator_engine_plugin_registry(capture_root=capture_root)
    assert "robot_eval_simulator_engine_plugins_not_ready" in registry["blockers"]
    assert "robot_eval_world_model_engine_plugins_not_ready" in registry["blockers"]
    assert registry["missing_simulator_plugin_local_inputs"]["isaac_lab_arena"]
    assert registry["missing_world_model_plugin_local_inputs"]["worldlabs_world_model"]
    _write_json(
        automation_root / "simulator_engine_plugin_registry.json",
        {
            "plugins": {
                "isaac_lab_arena": {
                    "adapter_contract_status": "ready",
                    "managed_execution_supported": True,
                    "inputs": {},
                }
            },
            "world_model_plugins": {
                "worldlabs_world_model": {
                    "adapter_contract_status": "ready",
                    "managed_execution_supported": True,
                    "inputs": {},
                }
            },
        },
    )
    missing_required_inputs = co._validate_simulator_engine_plugin_registry(capture_root=capture_root)
    assert "robot_eval_simulator_engine_plugin_registry_missing_required_input_keys" in missing_required_inputs["blockers"]
    assert "robot_eval_world_model_engine_plugin_registry_missing_required_input_keys" in missing_required_inputs["blockers"]


def test_auto_stage_robot_eval_and_inbox_runtime_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1"
    descriptor = capture_root / "capture_descriptor.json"
    _write_json(
        descriptor,
        {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "metadata": {"site_identity": {"site_id": "site-1"}},
        },
    )
    robot_eval_root = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(robot_eval_root / "task_cards.json", {"cards": [{"name": "bad"}, {"task_id": "task-1"}]})
    _write_json(robot_eval_root / "scenario_cards.json", {"cards": []})
    blocked = co._auto_stage_robot_eval_job_request(capture_root=capture_root, descriptor_path=descriptor)
    assert blocked["status"] == "blocked"
    assert "robot_eval_task_scenario_links_missing" in blocked["blockers"]

    _write_json(robot_eval_root / "scenario_cards.json", {"cards": [{"task_id": "task-1", "scenario_id": "scenario-1"}]})
    _write_json(robot_eval_root / "task_thresholds.json", {"thresholds": [{"task_id": "task-1"}]})
    _write_json(
        robot_eval_root / "scoring_methodology.json",
        {"metrics": [{"id": metric_id} for metric_id in co._STANDARD_ROBOT_EVAL_SCORECARD_METRICS]},
    )
    _write_json(robot_eval_root / "failure_taxonomy.json", {"failure_modes": []})
    empty_taxonomy = co._auto_stage_robot_eval_job_request(capture_root=capture_root, descriptor_path=descriptor)
    assert "robot_eval_failure_taxonomy_empty" in empty_taxonomy["blockers"]

    monkeypatch.setattr(co, "_load_cards", lambda path: [{"task_id": "task-1"}] if path.name == "task_cards.json" else [{"task_id": "task-1", "scenario_id": "scenario-1"}])
    monkeypatch.setattr(co, "_task_ids_from_thresholds", lambda _path: {"task-1"})
    monkeypatch.setattr(co, "_metric_ids_from_methodology", lambda _path: set(co._STANDARD_ROBOT_EVAL_SCORECARD_METRICS))
    monkeypatch.setattr(
        co,
        "_validate_scenario_variation_artifacts",
        lambda **_kwargs: {
            "blockers": [],
            "missing_scenario_family_task_ids": [],
            "missing_scenario_family_scenario_ids": [],
            "missing_scenario_family_variations_by_family": [],
            "missing_scenario_variation_names": [],
            "missing_scenario_variation_names_by_scenario": [],
        },
    )
    monkeypatch.setattr(co, "_failure_mode_ids_from_taxonomy", lambda _path: {"drop"})
    monkeypatch.setattr(
        co,
        "_validate_robot_eval_dataset_required_inputs",
        lambda **_kwargs: {
            "blockers": [],
            "missing_robot_eval_dataset_inputs": [],
            "robot_eval_dataset_input_artifacts": {},
        },
    )
    monkeypatch.setattr(
        co,
        "_validate_simulator_engine_plugin_registry",
        lambda **_kwargs: {
            "blockers": [],
            "simulator_plugin_count": 1,
            "world_model_plugin_count": 1,
            "missing_simulator_plugins": [],
            "missing_world_model_plugins": [],
            "unready_simulator_plugins": [],
            "unready_world_model_plugins": [],
            "missing_simulator_plugin_input_keys": {},
            "missing_world_model_plugin_input_keys": {},
            "missing_simulator_plugin_local_inputs": {},
            "missing_world_model_plugin_local_inputs": {},
        },
    )
    staged = co._auto_stage_robot_eval_job_request(capture_root=capture_root, descriptor_path=descriptor)
    restaged = co._auto_stage_robot_eval_job_request(capture_root=capture_root, descriptor_path=descriptor)
    assert staged["status"] == "staged"
    assert restaged["status"] == "already_staged"

    monkeypatch.setattr(
        co,
        "_auto_stage_robot_eval_job_request",
        lambda **_kwargs: {
            "status": "blocked",
            "inbox_dir": "inbox",
            "blockers": ["missing"],
        },
    )
    blocked_inbox = co._run_robot_eval_job_inbox_if_ready(
        tmp_path / "blocked" / "capture",
        auto_stage_task_eval=True,
        descriptor_path=descriptor,
    )
    assert blocked_inbox["status"] == "blocked_missing_task_eval_inputs"
    monkeypatch.setattr(co, "_auto_stage_robot_eval_job_request", lambda **_kwargs: {"status": "not_requested"})
    waiting_root = tmp_path / "waiting" / "capture"
    waiting = co._run_robot_eval_job_inbox_if_ready(waiting_root)
    assert waiting["status"] == "waiting_for_job_requests"
    processed_root = tmp_path / "processed" / "capture"
    inbox = processed_root / "pipeline" / "robot_eval_job_requests" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "job.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(co, "run_robot_eval_job_request_inbox", lambda **_kwargs: {"status": "processed", "processed_count": 1})
    processed = co._run_robot_eval_job_inbox_if_ready(processed_root)
    assert processed["status"] == "processed"
    assert processed["runtime"]["simulator_command_configured"] is False


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


def test_capture_orchestrator_cli_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "gcs"
    descriptor_uri = _descriptor_uri(root)
    original_for_capture = co.run_capture_pipeline_for_capture
    monkeypatch.setenv("GCS_ROOT", str(root))
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(co, "run_capture_pipeline", lambda **kwargs: calls.append(("descriptor", kwargs)))
    monkeypatch.setattr(co, "run_capture_pipeline_for_capture", lambda **kwargs: calls.append(("capture", kwargs)))

    assert co.main(["--descriptor-gcs-uri", descriptor_uri, "--lane", "qualification"]) == 0
    missing_uri = "gs://bucket/scenes/scene-2/captures/capture-2/capture_descriptor.json"
    assert co.main(
        [
            "--descriptor-gcs-uri",
            missing_uri,
            "--bucket",
            "bucket",
            "--scene-id",
            "scene-2",
            "--capture-id",
            "capture-2",
        ]
    ) == 0
    assert co.main(["--bucket", "bucket", "--scene-id", "scene-3", "--capture-id", "capture-3"]) == 0
    assert [kind for kind, _ in calls] == ["descriptor", "capture", "capture"]
    assert "completed" in capsys.readouterr().out

    monkeypatch.setattr(co, "run_capture_pipeline", lambda **_kwargs: (_ for _ in ()).throw(PipelineError("bad")))
    assert co.main(["--descriptor-gcs-uri", descriptor_uri]) == 1
    assert "FAILED: bad" in capsys.readouterr().out
    monkeypatch.setattr(
        co,
        "materialize_capture_bundle",
        lambda **_kwargs: {"descriptor_uri": descriptor_uri},
    )
    monkeypatch.setattr(co, "run_capture_pipeline_for_capture", original_for_capture)
    monkeypatch.setattr(co, "run_capture_pipeline", lambda **kwargs: {"status": "completed", **kwargs})
    wrapped = co.run_capture_pipeline_for_capture(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        config=co.PipelineConfig(gcs_root=root),
    )
    assert wrapped["descriptor_gcs_uri"] == descriptor_uri
    with pytest.raises(SystemExit):
        co.main([])
    guard = compile("raise SystemExit(main())", co.__file__, "exec").replace(co_firstlineno=1925)
    with pytest.raises(SystemExit) as exc_info:
        exec(guard, {"main": lambda: 0})
    assert exc_info.value.code == 0
