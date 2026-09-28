"""ADP-009D/day-21: distinct team requests reuse a scene without sharing a run."""
from pathlib import Path
import json

import pytest

from blueprint_pipeline.task_evaluation_configured_controls_progression import (
    TaskEvaluationConfiguredControlsProgressionError,
    stage_configured_controls_episode_preparation,
)
from tests.test_task_evaluation_configured_controls_progression import (
    _configured, _fake_materializer, _publisher, _runtime,
)


def _prepare(root: Path, run_id: str, *, output_root: Path | None = None):
    terminal, publication, revision = _configured()
    return stage_configured_controls_episode_preparation(
        terminal_result=terminal, publication_result=publication,
        configured_revision=revision, expected_production_commit="b" * 40,
        robot_mount_interface_path=root / "mount.json",
        scene_camera_calibration_path=root / "calibration.json",
        base_pose_candidate={}, cameras=[], runtime_binding=_runtime(),
        output_root=output_root or root / "runs" / run_id, publisher=_publisher,
        queue_root=root / "queue", submitted_by="robot-team-development-request",
        readiness_materializer=_fake_materializer, evaluation_run_id=run_id,
    )


def test_two_requests_keep_independent_queue_and_asset_names(tmp_path):
    first = _prepare(tmp_path, "team-run-one")
    second = _prepare(tmp_path, "team-run-two")
    a, b = (r["episode_preparation_request"] for r in (first, second))
    assert a["scene"] == b["scene"]
    assert a["task"] == b["task"]
    assert a["preparation_id"] != b["preparation_id"]
    assert a["run_id"] != b["run_id"]
    assert a["publication"]["input_namespace"] != b["publication"]["input_namespace"]
    assert a["robot"]["configuration"]["uri"] != b["robot"]["configuration"]["uri"]
    assert a["spend"] == b["spend"] == _runtime()["spend"]
    assert _prepare(tmp_path, "team-run-one") == first
    assert first["provider_mutation_performed"] is False
    assert first["native_construction_readback_required"] is True
    assert first["evaluation_run_id"] == "team-run-one"


def test_reused_output_directory_cannot_adopt_another_request(tmp_path):
    output = tmp_path / "shared-output"
    _prepare(tmp_path, "one", output_root=output)
    with pytest.raises(TaskEvaluationConfiguredControlsProgressionError, match="immutable_conflict"):
        _prepare(tmp_path, "two", output_root=output)


@pytest.mark.parametrize("run_id", ["", "../foreign", "x" * 193, 42])
def test_invalid_run_scope_is_rejected_before_materialization(tmp_path, run_id):
    with pytest.raises(TaskEvaluationConfiguredControlsProgressionError,
                       match="evaluation_run_id_invalid"):
        terminal, publication, revision = _configured()
        stage_configured_controls_episode_preparation(
            terminal_result=terminal, publication_result=publication,
            configured_revision=revision, expected_production_commit="b" * 40,
            robot_mount_interface_path=tmp_path / "mount.json",
            scene_camera_calibration_path=tmp_path / "calibration.json",
            base_pose_candidate={}, cameras=[], runtime_binding=_runtime(),
            output_root=tmp_path / "outputs", publisher=_publisher,
            queue_root=tmp_path / "queue", submitted_by="robot-team-development-request",
            readiness_materializer=_fake_materializer, evaluation_run_id=run_id,
        )
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("with_authority", [False, True])
def test_plans_for_same_scene_keep_independent_future_launches(tmp_path, monkeypatch, with_authority):
    from blueprint_pipeline.task_evaluation_configured_controls_plan import materialize_configured_controls_plan
    from blueprint_pipeline.configured_controls_plan_validation import read_configured_controls_plan
    from tests.test_task_evaluation_configured_controls_plan import (
        _bindings, _qualifying_source, SOURCE_LAUNCH_ID, TARGET_COMMIT,
    )
    _qualifying_source(monkeypatch)
    (tmp_path / "profiles").mkdir()
    inputs = dict(source_launch_id=SOURCE_LAUNCH_ID, launch_state_root=tmp_path / "launches",
        expected_production_commit=TARGET_COMMIT, submitted_by="robot-team-request",
        bindings=_bindings(tmp_path), plan_root=tmp_path / "plans", profile_dir=tmp_path / "profiles")
    plans = []
    for run_id in ("one", "two"):
        authority = {"evaluation_run_id":run_id, "source_launch_id":SOURCE_LAUNCH_ID,
            "source_profile_digest":"sha256:"+"a"*64, "configured_scene_revision_digest":"sha256:"+"b"*64,
            "scene_intent_digest":"sha256:"+"c"*64}
        extra = {"evaluation_authority":authority} if with_authority else {}
        result = materialize_configured_controls_plan(**inputs, evaluation_run_id=run_id, **extra)
        path = Path(result["plan_path"])
        plan = read_configured_controls_plan(path)
        assert plan["evaluation_run_id"] == run_id
        assert materialize_configured_controls_plan(**inputs, evaluation_run_id=run_id, **extra)["status"] == "replayed"
        if with_authority:
            assert plan["evaluation_authority"] == authority
        plans.append(plan)
    assert plans[0]["future_outputs"] != plans[1]["future_outputs"]
    assert len(list((tmp_path / "plans").glob("*.json"))) == 2
    assert plans[0]["source_launch_receipt_digest"] == plans[1]["source_launch_receipt_digest"]
    assert plans[0]["artifact_inventory"] == plans[1]["artifact_inventory"]


def test_handoff_uses_the_same_per_request_state_directory(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_policy_canary_handoff as handoff
    from blueprint_pipeline.configured_scene_run_identity import progression_directory
    secret = tmp_path / "credential"
    secret.write_text("test-only-credential")
    secret.chmod(0o600)
    observed = []
    monkeypatch.setattr(handoff, "advance_policy_canary_handoff", lambda **kw: observed.append(kw) or {})
    for run_id in ("one", "two"):
        handoff.advance_policy_canary_handoff_for_plan(
            plan={"source_launch_id": "shared-source", "expected_production_commit": "b" * 40,
                "evaluation_run_id": run_id, "profile_dir": str(tmp_path / "profiles")},
            progression_root=tmp_path / "progression", launch_state_root=tmp_path / "launches",
            episode_compilation_queue_root=tmp_path / "compilation", activation_intent_root=tmp_path / "intents",
            repo_root=tmp_path, webapp_secret_file=secret, webapp_endpoint="https://example.invalid/api",
            webapp_catalog_out=tmp_path / "catalog", notification_email="test@example.invalid",
            publisher_factory=lambda: None,
        )
    assert observed[0]["state_root"] != observed[1]["state_root"]
    for call in observed:
        assert call["state_root"].name == progression_directory("b" * 40, call["evaluation_run_id"])
        assert call["source_launch_id"] == "shared-source"


def test_controller_forwards_request_scope_and_keeps_each_progression_separate(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
    from blueprint_pipeline.configured_scene_run_identity import episode_namespace, progression_directory
    from tests.test_task_evaluation_configured_controls_progression_worker import _source, _plan, _seal_plan
    launches, _ = _source(tmp_path)
    original = json.loads(_plan(tmp_path).read_text())
    terminal, _, _ = worker._validate_source(launches / original["source_launch_id"])
    observed = []
    monkeypatch.setattr(worker, "stage_configured_controls_episode_preparation",
                        lambda **kw: observed.append(kw) or {"status": "episode_preparation_queued"})
    for run_id in ("one", "two"):
        plan = json.loads(json.dumps(original))
        plan["evaluation_run_id"] = run_id
        namespace = episode_namespace(terminal["run_id"], plan["expected_production_commit"],
                                      evaluation_run_id=run_id)
        for phase in ("construction", "controls"):
            plan["future_outputs"][phase]["expected_activation_id"] = f"{namespace}-episode-{phase}"
        _seal_plan(plan)
        path = tmp_path / f"plan-{run_id}.json"
        path.write_text(json.dumps(plan))
        worker.advance_configured_controls_plan(
            plan_path=path, launch_state_root=launches, progression_root=tmp_path / "progressions",
            preparation_queue_root=tmp_path / "preparations", activation_queue_root=tmp_path / "activations",
            publisher_factory=lambda: object(),
        )
    assert [row["evaluation_run_id"] for row in observed] == ["one", "two"]
    assert observed[0]["output_root"] != observed[1]["output_root"]
    for row in observed:
        assert row["output_root"].name == progression_directory(row["expected_production_commit"], row["evaluation_run_id"])
