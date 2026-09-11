import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_completed_placement_adoption as adoption
from blueprint_pipeline import task_evaluation_configured_controls_autostart as auto
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_completed_placement_rebinds_native_plan_without_new_model_or_search(tmp_path, monkeypatch):
    scene = {"scene": "fixture"}
    task = {"task": "move-object"}
    trajectory = {"trajectory_digest": "sha256:" + "a" * 64}
    revision = {"revision_digest": "sha256:" + "b" * 64}
    cameras = tmp_path / "old-cameras.json"
    cameras.write_text(json.dumps({"cameras": [{"pose": "fixed"}]}))
    universe = tmp_path / "old-universe.json"
    universe.write_text(json.dumps({"run_id": "retained-run"}))
    old = {
        "scene_binding_digest": canonical_digest(scene),
        "task_binding_digest": canonical_digest(task),
        "trajectory_digest": trajectory["trajectory_digest"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "native_construction_candidate_universe": {"path": str(universe)},
        "placement_agent_receipt_digest": "sha256:" + "c" * 64,
        "official_openai_cost_evidence": {"retained": True},
        "cpu_placement_checkpoint_binding_digest": "sha256:" + "d" * 64,
    }
    placement = {
        "accepted_pose": {"position_world_m": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]},
        "accepted_candidate_id": "selected",
    }
    files = {
        k: {"digest": "sha256:" + "e" * 64}
        for k in (
            "robot_asset_usd_path",
            "robot_mount_interface_path",
            "scene_camera_calibration_path",
        )
    }
    source = {
        "result": old,
        "inventory": {},
        "placement": placement,
        "intent": {"artifact_inventory": files},
        "plan": {"cameras_path": str(cameras)},
    }
    packet = {"source_result": {"digest": "sha256:" + "f" * 64}}
    monkeypatch.setattr(adoption, "validate_adoption", lambda value: source)
    seen = []

    def forbid(**kwargs):
        raise AssertionError("completed placement must not rerun")

    monkeypatch.setattr(auto, "run_robot_placement_cli", forbid)

    def native_universe(**kwargs):
        assert kwargs["run_id"] == "retained-run"
        seen.append("universe")
        return universe, {"inventory_digest": "sha256:" + "0" * 64, "candidates": [{}]}

    monkeypatch.setattr(auto, "_materialize_native_feedback_candidate_universe", native_universe)
    monkeypatch.setattr(auto, "_materialize_placement_aware_cameras", lambda **kwargs: cameras)

    def readiness(**kwargs):
        assert kwargs["placement_receipt"] is placement
        Path(kwargs["output_path"]).write_text("{}")
        seen.append("readiness")

    def plan(**kwargs):
        assert set(kwargs["bindings"]["phases"]) == {"destination", "construction", "controls"}
        assert kwargs["expected_production_commit"] == "1" * 40
        seen.append("plan")
        return {"plan_path": str(tmp_path / "new-plan.json"), "plan_digest": "sha256:" + "2" * 64}

    intent = {
        "completed_placement_adoption": packet,
        "expected_production_commit": "1" * 40,
        "artifact_inventory": files,
        "placement": {"candidate_inventory_cap": 24},
        "intent_digest": "sha256:" + "3" * 64,
        "phases": {"destination": {}, "construction": {}, "controls": {}},
        "profile_dir": str(tmp_path),
        "submitted_by": "fixture",
    }
    paths = {
        k: str(tmp_path / k)
        for k in (
            "cameras_path",
            "robot_mount_interface_path",
            "scene_camera_calibration_path",
            "runtime_binding_path",
        )
    }
    kwargs = dict(
        intent=intent,
        root=tmp_path,
        source_launch_id="source",
        launch_root=tmp_path,
        paths=paths,
        revision=revision,
        scene_binding=scene,
        task_binding=task,
        trajectory=trajectory,
        plan_root=tmp_path,
        readiness_materializer=readiness,
        plan_materializer=plan,
    )
    result = adoption.materialize(**kwargs)
    assert result["placement_agent_receipt_digest"] == old["placement_agent_receipt_digest"]
    assert (
        result["cpu_placement_checkpoint_binding_digest"]
        == old["cpu_placement_checkpoint_binding_digest"]
    )
    assert result["official_openai_cost_evidence"] == old["official_openai_cost_evidence"]
    assert result["placement_calls_reexecuted"] is False and seen == [
        "universe",
        "readiness",
        "plan",
    ]
    with pytest.raises(ValueError, match="scientific_binding_changed"):
        adoption.materialize(**{**kwargs, "task_binding": {"task": "different"}})


def test_native_submission_prevents_budget_retirement(tmp_path):
    config = {
        "scene_root": str(tmp_path / "owners"),
        "progression_root": str(tmp_path / "progression"),
        "launch_state_root": str(tmp_path / "launches"),
    }
    plan = {
        "source_launch_id": "source",
        "expected_production_commit": "a" * 40,
        "future_outputs": {"construction": {"expected_activation_id": "activation"}},
    }
    assert adoption.native_submission_absent(config=config, plan=plan)
    marker = (
        tmp_path
        / "progression"
        / "source"
        / ("franka-controls-" + "a" * 12)
        / "construction_activation_progression.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    assert not adoption.native_submission_absent(config=config, plan=plan)
    marker.unlink()
    (tmp_path / "launches" / "activation-auto-launch").mkdir(parents=True)
    assert not adoption.native_submission_absent(config=config, plan=plan)


def test_repeated_adoption_uses_original_checkpoint_without_new_model_file(tmp_path, monkeypatch):
    original = tmp_path / "accepted-checkpoint.json"
    original.write_text('{"original_intent": "first-model-call"}\n')
    reference = adoption._file(original)
    inherited = {"source_agent_checkpoint": reference}
    validated = []
    monkeypatch.setattr(adoption, "validate_adoption", lambda value: validated.append(value))
    result = {"completed_placement_adoption": inherited, "placement_calls_reexecuted": False}
    assert (
        adoption.checkpoint_reference(result=result, binding=tmp_path, token="new-intent")
        == reference
    )
    assert validated == [inherited]
    assert not (tmp_path / "agent-placement-checkpoint-new-intent.v1.json").exists()
    original.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="reference_changed"):
        adoption.checkpoint_reference(result=result, binding=tmp_path, token="new-intent")


def test_legacy_checkpoint_alias_is_byte_identical_idempotent_and_never_overwrites(
    tmp_path, monkeypatch
):
    binding = tmp_path / "source-launch" / "cpu-robot-binding"
    binding.mkdir(parents=True)
    original = tmp_path / "accepted.json"
    original.write_text('{ "original_intent": "accepted-model-call" }\n')
    reference = adoption._file(original)
    inherited = {"source_agent_checkpoint": reference, "source_launch_id": "source-launch"}
    intent = {"intent_digest": "sha256:" + "a" * 64, "completed_placement_adoption": inherited}
    intent_path = tmp_path / "intent.json"
    intent_path.write_text(json.dumps(intent))
    result = {
        "completed_placement_adoption": inherited,
        "placement_calls_reexecuted": False,
        "scene_binding_digest": "scene",
        "task_binding_digest": "task",
        "cpu_placement_checkpoint_binding_digest": "cpu",
    }
    result_path = auto._autostart_result_path(root=binding, intent_digest=intent["intent_digest"])
    result_path.write_text(json.dumps(result))
    monkeypatch.setattr(auto, "validate_configured_controls_autostart_intent", lambda value: value)
    monkeypatch.setattr(auto, "_validate_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(adoption, "validate_adoption", lambda value: None)
    first = adoption.materialize_legacy_checkpoint_alias(
        intent_path=intent_path, binding_root=binding
    )
    assert Path(first["target"]["path"]).read_bytes() == original.read_bytes()
    assert first["target"]["digest"] == reference["digest"]
    assert first["placement_calls_reexecuted"] is False
    assert (
        adoption.materialize_legacy_checkpoint_alias(intent_path=intent_path, binding_root=binding)[
            "status"
        ]
        == "already_present"
    )
    target = Path(first["target"]["path"])
    target.chmod(0o640)
    target.write_text('{"unrelated": true}')
    with pytest.raises(ValueError, match="checkpoint_alias_conflict"):
        adoption.materialize_legacy_checkpoint_alias(intent_path=intent_path, binding_root=binding)
    assert target.read_text() == '{"unrelated": true}'
