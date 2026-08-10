from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import episode_spec as ep


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_episode_spec_helper_edges() -> None:
    assert ep._string_list(None) == []
    assert ep._string_list("one") == ["one"]
    assert ep._string_list(7) == ["7"]
    assert ep._stable_slug("", fallback="task") == "task"
    assert ep._stable_slug("123 task", fallback="task") == "n_123_task"
    assert ep._float_list(["bad"], fallback=[1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert ep._task_category_from_text("pick a tote") == "pick_place"
    assert ep._task_category_from_text("inspect shelf") == "inspection_route"
    assert ep._task_category_from_text("drive route") == "navigation"
    assert ep._scene_class_task_hints("warehouse")[0]["task_id"] == "warehouse_tote_transfer"
    assert ep._missing_proof_labels(
        task={"anchor_accepted": True},
        robot_profile_from_request=True,
        frame={"scale_proven": True},
        scorecard={"isaac_usd_collision_verified": True, "portable_collider_glb_missing": True},
    ) == ["simulator_execution_not_run", "portable_collider_glb_missing"]
    assert ep._default_agent_proposals(generated_at="2026-06-21T00:00:00+00:00")[
        "status"
    ] == "not_requested"
    proposal = ep._proposal_from_hint(
        {
            "task_id": "task-1",
            "task_text": "Inspect bin",
            "source_context": {"workflow_name": "bin-check"},
        },
        index=0,
    )
    assert proposal["source_context"] == {"workflow_name": "bin-check"}


def test_episode_spec_hint_loaders_cover_capture_object_scene_and_hypothesis(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    pipeline_dir = capture_root / "pipeline"
    automation_dir = pipeline_dir / "simulation_automation"
    assert ep._capture_manifest_task_hints(capture_root) == []

    _write_json(capture_root / "raw" / "manifest.json", {"workflowName": "humanoid route"})
    route_hint = ep._capture_manifest_task_hints(capture_root)[0]
    assert route_hint["target_object_ids"] == ["selected_waypoint"]
    assert "humanoid" in route_hint["task_text"].lower()
    _write_json(capture_root / "raw" / "manifest.json", {"workflowName": "warehouse route"})
    route_hint = ep._capture_manifest_task_hints(capture_root)[0]
    assert route_hint["task_text"] == "Navigate from validated start zone to selected waypoint"

    _write_json(capture_root / "raw" / "manifest.json", {"task_steps": ["Pick bin", "Place tote"]})
    steps_hint = ep._capture_manifest_task_hints(capture_root)[0]
    assert steps_hint["task_text"].startswith("Execute capture-described task")

    _write_json(capture_root / "raw" / "manifest.json", {"zone": "Dock"})
    zone_hint = ep._capture_manifest_task_hints(capture_root)[0]
    assert zone_hint["task_text"] == "Execute task in Dock"

    assert ep._object_label_task_hints(pipeline_dir) == []
    _write_json(
        pipeline_dir / "evaluation_prep" / "object_geometry_manifest.json",
        {"objects": ["bad", {"label": ""}, {"object_id": "bin-1", "label": "Bin A"}]},
    )
    object_hints = ep._object_label_task_hints(pipeline_dir)
    assert object_hints == [
        {
            "task_id": "inspect_bin_1",
            "task_text": "Inspect or approach Bin A",
            "task_category": "object_inspection",
            "target_object_ids": ["bin-1"],
            "source": "evaluation_prep/object_geometry_manifest.json",
        }
    ]

    assert ep._scene_asset_task_hints(automation_dir) == []
    _write_json(
        automation_dir / "scene_asset_inspection.json",
        {
            "assets": [
                "bad",
                {
                    "semantic_hints": [
                        "bad",
                        {"label": ""},
                        {"label": "floor"},
                        {"label": "Bin Shelf"},
                    ]
                },
            ]
        },
    )
    scene_hints = ep._scene_asset_task_hints(automation_dir)
    assert scene_hints[0]["task_category"] == "pick_place"

    assert ep._task_hypothesis_hints(capture_root) == []
    _write_json(
        capture_root / "raw" / "task_hypothesis.json",
        {
            "tasks": [
                {"task_id": "task-a", "task_text": "Scan shelf", "task_category": "inspection"},
                "Move to waypoint",
            ]
        },
    )
    hypothesis_hints = ep._task_hypothesis_hints(capture_root)
    assert [item["task_id"] for item in hypothesis_hints] == ["task-a", "task_hypothesis_2"]


def test_episode_spec_task_scenario_profile_loaders_and_default_proposals(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    pipeline_dir = capture_root / "pipeline"
    automation_dir = pipeline_dir / "simulation_automation"
    generated_at = "2026-06-21T00:00:00+00:00"

    default_manifest = ep.build_task_anchor_proposals(
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        automation_dir=automation_dir,
        generated_at=generated_at,
    )
    assert default_manifest["proposals"][0]["source"] == "deterministic_default"
    assert default_manifest["input_notes"] == []

    _write_json(capture_root / "capture_descriptor.json", {"site_type": "aquarium touch tank"})
    unrecognized_site_type_manifest = ep.build_task_anchor_proposals(
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        automation_dir=automation_dir,
        generated_at=generated_at,
    )
    assert unrecognized_site_type_manifest["input_notes"] == [
        {
            "code": "site_type_unrecognized",
            "detail": "No deterministic task template matched the capture site type text; generic/object-grounded task proposals remain review-only.",
            "site_type_text": "aquarium touch tank",
        }
    ]
    (capture_root / "capture_descriptor.json").unlink()

    _write_json(
        capture_root / "raw" / "task_hypothesis.json",
        {"tasks": [{"task_id": "dup-task", "task_text": "Inspect A"}, {"task_id": "dup-task", "task_text": "Inspect B"}]},
    )
    deduped = ep.build_task_anchor_proposals(
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        automation_dir=automation_dir,
        generated_at=generated_at,
    )
    assert [proposal["task_id"] for proposal in deduped["proposals"]] == ["dup_task"]

    _write_json(
        automation_dir / "task_anchor_proposal_manifest.json",
        {"proposals": ["bad", {"task_id": "proposal-task", "target_object_ids": ["bin"]}]},
    )
    proposal_tasks = ep._tasks_from_proposals(automation_dir)
    assert proposal_tasks[0]["task_id"] == "proposal-task"

    _write_json(
        pipeline_dir / "evaluation_prep" / "task_anchor_manifest.json",
        {"tasks": ["bad", {"id": "accepted-task", "accepted": True, "target_object_ids": "box"}]},
    )
    anchored_tasks = ep._load_tasks(pipeline_dir, automation_dir)
    assert anchored_tasks[0]["task_id"] == "accepted-task"
    assert anchored_tasks[0]["target_object_ids"] == ["box"]

    (pipeline_dir / "evaluation_prep" / "task_anchor_manifest.json").unlink()
    _write_json(
        pipeline_dir / "robot_eval_dataset" / "task_cards.json",
        {"cards": [{"task_id": "card-task", "task_statement": "Check a shelf"}]},
    )
    assert ep._load_tasks(pipeline_dir, automation_dir)[0]["anchor_source"] == (
        "robot_eval_dataset/task_cards.json"
    )

    (pipeline_dir / "robot_eval_dataset" / "task_cards.json").unlink()
    assert ep._load_tasks(pipeline_dir, automation_dir)[0]["anchor_source"] == (
        "simulation_automation/task_anchor_proposal_manifest.json"
    )
    (automation_dir / "task_anchor_proposal_manifest.json").unlink()
    assert ep._load_tasks(pipeline_dir, automation_dir)[0]["anchor_source"] == "deterministic_default"

    assert ep._load_scenarios(pipeline_dir, [{"task_id": "task"}])[0]["scenario_source"] == (
        "deterministic_default_capture_observed_layout"
    )
    _write_json(
        pipeline_dir / "robot_eval_dataset" / "scenario_cards.json",
        {"cards": ["bad", {"scenario_id": "scenario-1", "task_id": "task"}]},
    )
    assert ep._load_scenarios(pipeline_dir, [{"task_id": "task"}]) == [
        {
            "scenario_id": "scenario-1",
            "task_id": "task",
            "robot_profile_id": "",
            "scenario_source": "robot_eval_dataset/scenario_cards.json",
        }
    ]

    profiles, from_request = ep._robot_profiles(pipeline_dir)
    assert from_request is False
    assert profiles[0]["source"] == "deterministic_default_profile"
    _write_json(
        pipeline_dir / "evaluation_prep" / "site_world_spec.json",
        {"robot_profiles": ["bad", {"id": "robot-a", "label": "Robot A"}]},
    )
    profiles, from_request = ep._robot_profiles(pipeline_dir)
    assert from_request is True
    assert profiles == [
        {
            "id": "robot-a",
            "label": "Robot A",
            "robot_profile_id": "robot-a",
            "source": "site_world_or_hosted_session_runtime_manifest",
        }
    ]


def test_build_episode_specs_runs_preflight_fallback_and_fake_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    automation_dir = capture_root / "pipeline" / "simulation_automation"

    def fake_preflight(*, capture_root: Path) -> dict[str, object]:
        _write_json(
            Path(capture_root) / "pipeline" / "simulation_automation" / "scene_frame_estimate.json",
            {"frame": {"bounds": {"min": ["bad"], "max": [2, 3, 1]}, "floor_z_estimate": "bad"}},
        )
        return {"status": "written"}

    monkeypatch.setattr(ep, "build_scene_asset_preflight", fake_preflight)

    result = ep.build_episode_specs(
        capture_root=capture_root,
        agent_adapter=ep.FakeEpisodeSpecAgentAdapter(adapter_name="fixture-agent"),
    )

    episode_spec = json.loads(Path(result["episode_spec_path"]).read_text(encoding="utf-8"))
    proposals = json.loads(
        (automation_dir / "agent_episode_spec_proposals.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "compiled_review_required"
    assert episode_spec["episodes"][0]["robot_spawn_pose"]["xyz"][2] == 0.05
    assert proposals["adapter"] == "fixture-agent"
    assert proposals["proof_booleans_mutable_by_agent"] is False


def test_episode_spec_main_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[object] = []

    def fake_build_episode_specs(*, capture_root: str, agent_adapter: object) -> dict[str, object]:
        calls.append(agent_adapter)
        return {
            "episode_spec_path": str(tmp_path / "episode_spec.v1.json"),
            "status": "compiled",
        }

    monkeypatch.setattr(ep, "build_episode_specs", fake_build_episode_specs)
    assert ep.main(["--capture-root", str(tmp_path / "capture"), "--agent-mode", "fake"]) == 0
    assert isinstance(calls[0], ep.FakeEpisodeSpecAgentAdapter)
    assert "status=compiled" in capsys.readouterr().out

    def failing_build_episode_specs(*, capture_root: str, agent_adapter: object) -> dict[str, object]:
        raise ep.PipelineError("bad capture")

    monkeypatch.setattr(ep, "build_episode_specs", failing_build_episode_specs)
    assert ep.main(["--capture-root", str(tmp_path / "capture")]) == 1
    assert "FAILED: bad capture" in capsys.readouterr().out


def test_an_articulated_task_is_not_classified_as_navigation() -> None:
    """'Open the refrigerator door' is a task family, not a walk to a waypoint."""

    from blueprint_pipeline.episode_spec import _task_category_from_text

    assert (
        _task_category_from_text("Open the upper refrigerator door to 45 degrees")
        == "articulated_open_close"
    )
    assert _task_category_from_text("Pull the drawer open") == "articulated_open_close"
    # the existing families are unchanged
    assert _task_category_from_text("Pick the can and place it") == "pick_place"
    assert _task_category_from_text("Inspect the bench target zone") == "inspection_route"
    assert _task_category_from_text("Drive to the loading bay") == "navigation"
    # pre-existing precedence, recorded rather than silently changed here:
    # "bin" is a pick_place token, so an inspection phrased around bins already
    # classified as pick_place before articulation was added.
    assert _task_category_from_text("Inspect the labeled bins") == "pick_place"
