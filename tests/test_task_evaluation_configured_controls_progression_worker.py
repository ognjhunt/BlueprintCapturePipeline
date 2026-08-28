from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, *, provider_zero: bool = True) -> tuple[Path, Path]:
    launch_id = "scene-839873-qualifying"
    run_root = tmp_path / "launch-runs" / launch_id
    terminal_path = tmp_path / "allocator" / "result.json"
    publication_path = tmp_path / "allocator" / "publication.json"
    revision_path = tmp_path / "allocator" / "revision.json"
    _write(publication_path, {"schema_version": "test.publication.v1"})
    _write(revision_path, {"schema_version": "test.revision.v1"})
    _write(
        terminal_path,
        {
            "schema_version": "task_evaluation_scene_configuration_vast_result.v1",
            "publication_result_path": str(publication_path),
            "configured_scene_revision_path": str(revision_path),
        },
    )
    receipt: dict[str, object] = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "completed",
        "launch_id": launch_id,
        "run_id": "scene-839873",
        "request_digest": "sha256:" + "1" * 64,
        "launch_profile_digest": "sha256:" + "2" * 64,
        "terminal_evidence": {
            "status": "passed",
            "result": {
                "path": str(terminal_path),
                "exists": True,
                "digest": _digest(terminal_path),
            },
            "scene_configuration": {"configuration_completed": True},
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(run_root / "launch_receipt.json", receipt)
    sync: dict[str, object] = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "launch_id": launch_id,
        "run_id": receipt["run_id"],
        "request_digest": receipt["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "attempt_number": 1,
        "attempted_at": "2026-08-28T20:00:00+00:00",
        "provider_mutation_performed": False,
        "response": {
            "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
            "status": "completed",
            "already_exists": False,
            "launch_id": launch_id,
            "run_id": receipt["run_id"],
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "sync_result_digest": "",
    }
    sync["sync_result_digest"] = canonical_digest(sync, digest_field="sync_result_digest")
    _write(run_root / "webapp_sync_succeeded.json", sync)
    if provider_zero:
        zero: dict[str, object] = {
            "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
            "status": "provider_zero_confirmed",
            "launch_id": launch_id,
            "run_id": receipt["run_id"],
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
            "launch_profile_digest": receipt["launch_profile_digest"],
            "provider_zero_verified": True,
            "continuing_spend_from_this_run": False,
            "allocator_invoked": False,
            "provider_mutation_performed": False,
            "automatic_retry_performed": False,
            "blockers": [],
            "provider_zero_receipt_digest": "",
        }
        zero["provider_zero_receipt_digest"] = canonical_digest(
            zero, digest_field="provider_zero_receipt_digest"
        )
        _write(run_root / "post_teardown_provider_zero_receipt.json", zero)
    return run_root.parent, terminal_path


def _plan(tmp_path: Path) -> Path:
    for name, value in {
        "base.json": {"candidate": "franka-behind-push"},
        "cameras.json": {"cameras": [{"id": "external"}]},
        "runtime.json": {"runtime": {}, "execution_adapter": {}, "spend": {}},
        "mount.json": {"mount": "franka"},
        "calibration.json": {"camera": "external"},
    }.items():
        _write(tmp_path / "inputs" / name, value)
    plan: dict[str, object] = {
        "schema_version": worker.PLAN_SCHEMA_VERSION,
        "enabled": True,
        "source_launch_id": "scene-839873-qualifying",
        "expected_production_commit": "a" * 40,
        "submitted_by": "configured-controls-progression",
        "robot_mount_interface_path": str(tmp_path / "inputs" / "mount.json"),
        "scene_camera_calibration_path": str(tmp_path / "inputs" / "calibration.json"),
        "base_pose_candidate_path": str(tmp_path / "inputs" / "base.json"),
        "cameras_path": str(tmp_path / "inputs" / "cameras.json"),
        "runtime_binding_path": str(tmp_path / "inputs" / "runtime.json"),
        "phases": {"construction": {}, "controls": {}},
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    path = tmp_path / "plans" / "scene-839873.json"
    _write(path, plan)
    return path


def test_qualifying_terminal_and_post_teardown_zero_auto_queue_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root, _ = _source(tmp_path)
    plan = _plan(tmp_path)
    observed: dict[str, object] = {}

    def stage(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "episode_preparation_queued"}

    monkeypatch.setattr(worker, "stage_configured_controls_episode_preparation", stage)
    result = worker.advance_configured_controls_plan(
        plan_path=plan,
        launch_state_root=launch_root,
        progression_root=tmp_path / "progressions",
        preparation_queue_root=tmp_path / "preparations",
        activation_queue_root=tmp_path / "activations",
        publisher_factory=lambda: object(),
    )

    assert result["status"] == "episode_preparation_queued"
    assert observed["expected_production_commit"] == "a" * 40
    assert observed["queue_root"] == tmp_path / "preparations"
    assert observed["submitted_by"] == "configured-controls-progression"


def test_missing_post_teardown_global_zero_fails_before_queue_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root, _ = _source(tmp_path, provider_zero=False)
    plan = _plan(tmp_path)
    called = False

    def stage(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(worker, "stage_configured_controls_episode_preparation", stage)
    with pytest.raises(
        worker.TaskEvaluationConfiguredControlsProgressionWorkerError,
        match="configured_controls_worker_post_teardown_provider_zero_missing",
    ):
        worker.advance_configured_controls_plan(
            plan_path=plan,
            launch_state_root=launch_root,
            progression_root=tmp_path / "progressions",
            preparation_queue_root=tmp_path / "preparations",
            activation_queue_root=tmp_path / "activations",
            publisher_factory=lambda: object(),
        )
    assert called is False


def _sealed_progression(status: str, **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "task_evaluation_configured_controls_progression.v1",
        "status": status,
        **extra,
        "progression_digest": "",
    }
    value["progression_digest"] = canonical_digest(
        value, digest_field="progression_digest"
    )
    return value


def test_timer_advances_completed_preparation_into_construction_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for name in ("release_window", "lineage", "authorization"):
        path = tmp_path / "inputs" / f"{name}.json"
        _write(path, {"kind": name})
        plan["phases"]["construction"][f"{name}_path"] = str(path)
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    _write(plan_path, plan)
    state = tmp_path / "progressions" / plan["source_launch_id"]
    base = _sealed_progression(
        "episode_preparation_queued",
        episode_preparation_request={"preparation_id": "prep-1"},
    )
    _write(state / "configured_controls_progression.v1.json", base)
    prep_root = tmp_path / "preparations"
    _write(prep_root / "identities" / "prep-1.json", {"identity": "prep-1"})
    _write(prep_root / "results" / "prep-1-a.json", {"status": "prepared"})
    seen: dict[str, object] = {}

    def stage(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return _sealed_progression("construction_activation_queued")

    monkeypatch.setattr(worker, "stage_configured_controls_activation", stage)
    result = worker.advance_configured_controls_plan(
        plan_path=plan_path,
        launch_state_root=tmp_path / "unused-launch-runs",
        progression_root=tmp_path / "progressions",
        preparation_queue_root=prep_root,
        activation_queue_root=tmp_path / "activations",
        publisher_factory=lambda: object(),
    )
    assert result["status"] == "construction_activation_queued"
    assert seen["lane"] == "native_task_arena_construction"
    assert seen["queue_root"] == tmp_path / "activations"
    assert (state / "construction_activation_progression.json").is_file()


def test_paid_transition_uses_only_injected_webapp_submitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    activation_result = tmp_path / "inputs" / "construction-activation-result.json"
    profile = tmp_path / "inputs" / "construction-profile.json"
    authority = tmp_path / "inputs" / "construction-launch-authority.json"
    for path in (activation_result, profile, authority):
        _write(path, {"path": path.name})
    plan["phases"]["construction"] = {
        "activation_result_path": str(activation_result),
        "profile_path": str(profile),
        "launch_authority_path": str(authority),
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    _write(plan_path, plan)
    state = tmp_path / "progressions" / plan["source_launch_id"]
    _write(
        state / "configured_controls_progression.v1.json",
        _sealed_progression(
            "episode_preparation_queued",
            episode_preparation_request={"preparation_id": "prep-1"},
        ),
    )
    construction = _sealed_progression("construction_activation_queued")
    _write(state / "construction_activation_progression.json", construction)
    prep_root = tmp_path / "preparations"
    _write(prep_root / "identities" / "prep-1.json", {"identity": "prep-1"})
    _write(prep_root / "results" / "prep-1-a.json", {"status": "prepared"})
    observed: dict[str, object] = {}

    def submit_progression(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return _sealed_progression("construction_launch_queued")

    def webapp_submitter(value: dict[str, object]) -> dict[str, object]:
        return value
    monkeypatch.setattr(worker, "submit_authorized_progression_launch", submit_progression)
    result = worker.advance_configured_controls_plan(
        plan_path=plan_path,
        launch_state_root=tmp_path / "unused-launch-runs",
        progression_root=tmp_path / "progressions",
        preparation_queue_root=prep_root,
        activation_queue_root=tmp_path / "activations",
        publisher_factory=lambda: object(),
        submitter=webapp_submitter,
    )
    assert result["status"] == "construction_launch_queued"
    assert observed["submitter"] is webapp_submitter
    assert (state / "construction_launch_progression.json").is_file()


def test_systemd_worker_is_separate_from_reconciler_and_never_calls_allocator() -> None:
    service = Path(
        "deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service"
    ).read_text(encoding="utf-8")
    reconciler = Path(
        "deploy/systemd/blueprint-task-evaluation-launch-reconciler.service"
    ).read_text(encoding="utf-8")
    assert "task_evaluation_configured_controls_progression_worker" in service
    assert "post_teardown" not in reconciler
    assert "paid_resource_allocator" not in service
    assert "vast_provider_adapter" not in service
    assert "--webapp-secret-file" in service
    assert "BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_PLAN_ROOT" in service
