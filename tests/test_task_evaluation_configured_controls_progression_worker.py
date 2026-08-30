from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_configured_controls_autostart as autostart
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker


SOURCE_CONFIGURATION_COMMIT = "c" * 40


def _dynamic_window_inputs(tmp_path: Path) -> dict[str, object]:
    reference = {
        "uri": "s3://blueprint-production-inputs/lineage.json",
        "digest": "sha256:" + "4" * 64,
        "size_bytes": 1,
    }
    request = {
        "run_id": "scene-839873-franka-episode",
        "team_namespace": "blueprint-adp",
        "preparation_id": "prep-scene-839873",
        "spend": {
            "provider_allowlist": ["vast"],
            "hard_cap_usd": 1.0,
        },
    }
    state: dict[str, object] = {
        "schema_version": "task_evaluation_configured_controls_progression.v1",
        "status": "episode_preparation_queued",
        "configured_scene_revision_digest": "sha256:" + "5" * 64,
        "expected_production_commit": "a" * 40,
        "episode_preparation_request": request,
        "episode_preparation_request_digest": "sha256:" + "6" * 64,
        "progression_digest": "",
    }
    state["progression_digest"] = canonical_digest(
        state, digest_field="progression_digest"
    )
    preparation: dict[str, object] = {
        "schema_version": "task_evaluation_launch_preparation_result.v1",
        "status": "queued_for_production_episode_compilation",
        "run_mode": "episode_evaluation",
        "configured_scene_revision_digest": state[
            "configured_scene_revision_digest"
        ],
        "automatic_progression_required": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    preparation["result_digest"] = canonical_digest(
        preparation, digest_field="result_digest"
    )
    template: dict[str, object] = {
        "schema_version": "task_evaluation_configured_controls_release_window_template.v1",
        "status": "authorized_for_dynamic_release",
        "team_namespace": "blueprint-adp",
        "expected_production_commit": "a" * 40,
        "allowed_mutations": [
            "profile_publication",
            "catalog_synchronization",
            "standing_authorization",
        ],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": 1.0,
        "valid_for_seconds": 60,
        "released_by": "configured-controls-coordinator",
        "release_reference": "ADP-009D Day-28 continuation",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
        "template_digest": "",
    }
    template["template_digest"] = canonical_digest(
        template, digest_field="template_digest"
    )
    template_path = tmp_path / "release-window-template.json"
    _write(template_path, template)
    return {
        "state": state,
        "preparation": preparation,
        "phase": {"release_window_template_path": str(template_path)},
        "lineage": {
            "kind": "initial_project",
            "project_spend_reconciliation": reference,
            "initial_provider_zero": reference,
        },
        "authorization": {
            "reference": "ADP-009D Day-28",
            "authorized_by": "blueprint-owner",
            "authorized_on": "2026-08-29T00:00:00+00:00",
            "standing_authorization_expires_at": "2026-09-02T00:00:00+00:00",
            "profile_revision": "scene-839873-r1",
        },
    }


def _window_publisher(path: Path, object_name: str) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/{object_name}",
        "digest": _digest(path),
        "size_bytes": path.stat().st_size,
        "full_byte_service_account_readback_passed": True,
        "readback_digest": _digest(path),
        "readback_size_bytes": path.stat().st_size,
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_dynamic_window_retry_reopens_same_bytes_after_publish_failure(
    tmp_path: Path,
) -> None:
    values = _dynamic_window_inputs(tmp_path)
    observed: list[bytes] = []

    def flaky_publisher(*, path: Path, object_name: str) -> dict[str, object]:
        observed.append(path.read_bytes())
        if len(observed) == 1:
            raise RuntimeError("transient-object-store-failure")
        return _window_publisher(path, object_name)

    kwargs = {
        **values,
        "lane": "native_task_arena_construction",
        "root": tmp_path / "state",
        "publisher": flaky_publisher,
        "now": datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc),
    }
    with pytest.raises(RuntimeError, match="transient-object-store-failure"):
        worker._materialize_phase_release_window(**kwargs)
    result = worker._materialize_phase_release_window(
        **{
            **kwargs,
            "now": kwargs["now"] + timedelta(seconds=1),
        }
    )

    assert result["digest"] == "sha256:" + hashlib.sha256(observed[1]).hexdigest()
    assert observed[0] == observed[1]
    assert len(list((tmp_path / "state/release-window-attempts").rglob("window-*.json"))) == 1


def test_expired_dynamic_window_creates_versioned_refresh(tmp_path: Path) -> None:
    values = _dynamic_window_inputs(tmp_path)
    first_now = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    common = {
        **values,
        "lane": "native_task_arena_construction",
        "root": tmp_path / "state",
        "publisher": _window_publisher,
    }
    first = worker._materialize_phase_release_window(now=first_now, **common)
    second = worker._materialize_phase_release_window(
        now=first_now + timedelta(minutes=2), **common
    )

    assert first["digest"] != second["digest"]
    assert len(list((tmp_path / "state/release-window-attempts").rglob("window-*.json"))) == 2


def test_one_shot_adoption_registry_targets_only_its_legacy_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_id = "scene-839873-2deff449-r1"
    launch_root = tmp_path / "launch-runs"
    run_root = launch_root / launch_id
    _write(
        run_root / "launch_profile.json",
        {
            "immutable_inputs": [],
            "task_evaluation_run": {
                "team_namespace": "blueprint-adp",
                "scene_id": "interiorgs-839873",
                "task_id": "scene-839873-mug-planar-push",
            },
        },
    )
    _write(run_root / "launch_receipt.json", {"status": "completed"})
    _write(run_root / "webapp_sync_succeeded.json", {"status": "succeeded"})
    _write(
        run_root / "post_teardown_provider_zero_receipt.json",
        {"status": "provider_zero_confirmed"},
    )
    intent_root = tmp_path / "intents"
    adoption_path = intent_root / autostart.configured_controls_autostart_adoption_registry_name(
        team_namespace="blueprint-adp",
        scene_id="interiorgs-839873",
        task_id="scene-839873-mug-planar-push",
        source_launch_id=launch_id,
    )
    _write(adoption_path, {"schema_version": "test.adoption.v1"})
    automatic_path = intent_root / autostart.configured_controls_autostart_registry_name(
        team_namespace="blueprint-adp",
        scene_id="interiorgs-839873",
        task_id="scene-839873-mug-planar-push",
    )
    _write(automatic_path, {"schema_version": "test.automatic.v1"})
    adoption = {
        "configuration_adoption": {
            "mode": "explicit_terminal_adoption",
            "source_launch_id": launch_id,
        }
    }
    monkeypatch.setattr(
        autostart,
        "validate_configured_controls_autostart_intent",
        lambda _value: adoption,
    )
    observed: dict[str, object] = {}

    def materialize(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "status": "agent_binding_accepted_plan_materialized",
            "selected_candidate_id": "candidate-0001",
            "plan_digest": "sha256:" + "8" * 64,
        }

    monkeypatch.setattr(
        autostart, "materialize_configured_controls_autostart", materialize
    )
    report = worker.process_plans(
        plan_root=tmp_path / "plans",
        autostart_intent_root=intent_root,
        launch_state_root=launch_root,
        progression_root=tmp_path / "progression",
        preparation_queue_root=tmp_path / "preparations",
        activation_queue_root=tmp_path / "activations",
    )

    assert report["status"] == "completed"
    assert observed["source_launch_id"] == launch_id
    assert observed["intent_path_override"] == adoption_path
    assert observed["intent_path_override"] != automatic_path


def _seal_plan(plan: dict[str, object]) -> None:
    paths: dict[str, Path] = {}

    def collect(value: object, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if key.endswith("_path") and isinstance(child, str):
                paths[name] = Path(child)
            elif key not in {"artifact_inventory", "future_outputs"}:
                collect(child, name)

    collect(plan)
    plan["artifact_inventory"] = {
        name: {
            "path": str(path),
            "digest": _digest(path),
            "size_bytes": path.stat().st_size,
            "mode": f"{path.stat().st_mode & 0o777:04o}",
        }
        for name, path in sorted(paths.items())
    }
    plan.setdefault(
        "future_outputs",
        {
            "construction": {
                "expected_activation_id": "future-construction",
            },
            "controls": {
                "expected_activation_id": "future-controls",
            },
        },
    )
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")


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
            "run_id": "scene-839873-configuration",
            "publication_result_path": str(publication_path),
            "configured_scene_revision_path": str(revision_path),
        },
    )
    receipt: dict[str, object] = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "completed",
        "launch_id": launch_id,
        "source_commit": SOURCE_CONFIGURATION_COMMIT,
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
        "source_launch_receipt_digest": "sha256:" + "0" * 64,
        "source_configuration_commit": SOURCE_CONFIGURATION_COMMIT,
        "expected_production_commit": "a" * 40,
        "submitted_by": "configured-controls-progression",
        "profile_dir": str(tmp_path / "profiles"),
        "robot_mount_interface_path": str(tmp_path / "inputs" / "mount.json"),
        "scene_camera_calibration_path": str(tmp_path / "inputs" / "calibration.json"),
        "base_pose_candidate_path": str(tmp_path / "inputs" / "base.json"),
        "cameras_path": str(tmp_path / "inputs" / "cameras.json"),
        "runtime_binding_path": str(tmp_path / "inputs" / "runtime.json"),
        "phases": {"construction": {}, "controls": {}},
        "plan_digest": "",
    }
    for phase in ("construction", "controls"):
        for name in (
            "release_window_template_path",
            "authorization_path",
            "launch_authority_path",
        ):
            path = tmp_path / "inputs" / phase / f"{name}.json"
            _write(path, {"kind": f"{phase}-{name}"})
            plan["phases"][phase][name] = str(path)
    lineage_path = tmp_path / "inputs" / "construction" / "lineage_path.json"
    _write(lineage_path, {"kind": "initial_project"})
    plan["phases"]["construction"]["lineage_path"] = str(lineage_path)
    (tmp_path / "profiles").mkdir(exist_ok=True)
    receipt_path = (
        tmp_path
        / "launch-runs"
        / plan["source_launch_id"]
        / "launch_receipt.json"
    )
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        plan["source_launch_receipt_digest"] = receipt["receipt_digest"]
    _seal_plan(plan)
    if receipt_path.is_file():
        terminal_path = Path(receipt["terminal_evidence"]["result"]["path"])
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        namespace = (
            f"{terminal['run_id']}-franka-controls-"
            f"{str(plan['expected_production_commit'])[:12]}-episode"
        )
        for phase in ("construction", "controls"):
            plan["future_outputs"][phase]["expected_activation_id"] = (
                f"{namespace}-{phase}"
            )
        plan["plan_digest"] = canonical_digest(
            plan, digest_field="plan_digest"
        )
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


def test_source_configuration_commit_mismatch_fails_before_queue_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root, _ = _source(tmp_path)
    plan_path = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["source_configuration_commit"] = "d" * 40
    _seal_plan(plan)
    _write(plan_path, plan)
    called = False

    def stage(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        worker, "stage_configured_controls_episode_preparation", stage
    )
    with pytest.raises(
        worker.TaskEvaluationConfiguredControlsProgressionWorkerError,
        match="configured_controls_worker_source_receipt_mismatch",
    ):
        worker.advance_configured_controls_plan(
            plan_path=plan_path,
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
    for name in ("release_window_template", "lineage", "authorization"):
        path = tmp_path / "inputs" / f"{name}.json"
        _write(path, {"kind": name})
        plan["phases"]["construction"][f"{name}_path"] = str(path)
    _seal_plan(plan)
    _write(plan_path, plan)
    state = (
        tmp_path
        / "progressions"
        / plan["source_launch_id"]
        / f"franka-controls-{str(plan['expected_production_commit'])[:12]}"
    )
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
    monkeypatch.setattr(
        worker,
        "_materialize_phase_release_window",
        lambda **_: {
            "uri": "s3://configured-controls/window.json",
            "digest": "sha256:" + "7" * 64,
            "size_bytes": 1,
        },
    )
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


def test_progression_state_is_scoped_to_the_expected_production_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redeployed commit must derive its own progression state.

    Reusing the previous commit's sealed receipt carries its preparation id
    forward, and preparation results are immutable, so a launch that blocked
    under the old commit can never be answered under the new one.
    """

    plan_path = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for name in ("release_window_template", "lineage", "authorization"):
        path = tmp_path / "inputs" / f"{name}.json"
        _write(path, {"kind": name})
        plan["phases"]["construction"][f"{name}_path"] = str(path)
    _seal_plan(plan)
    _write(plan_path, plan)

    launch = tmp_path / "progressions" / plan["source_launch_id"]
    _write(
        launch / "configured_controls_progression.v1.json",
        _sealed_progression(
            "episode_preparation_queued",
            episode_preparation_request={"preparation_id": "prep-previous-commit"},
        ),
    )
    scoped = launch / (
        f"franka-controls-{str(plan['expected_production_commit'])[:12]}"
    )
    _write(
        scoped / "configured_controls_progression.v1.json",
        _sealed_progression(
            "episode_preparation_queued",
            episode_preparation_request={"preparation_id": "prep-this-commit"},
        ),
    )

    prep_root = tmp_path / "preparations"
    for prep in ("prep-previous-commit", "prep-this-commit"):
        _write(prep_root / "identities" / f"{prep}.json", {"identity": prep})
    _write(prep_root / "results" / "prep-previous-commit-a.json", {"status": "blocked"})
    _write(prep_root / "results" / "prep-this-commit-a.json", {"status": "prepared"})

    monkeypatch.setattr(
        worker,
        "stage_configured_controls_activation",
        lambda **_: _sealed_progression("construction_activation_queued"),
    )
    monkeypatch.setattr(
        worker,
        "_materialize_phase_release_window",
        lambda **_: {
            "uri": "s3://configured-controls/window.json",
            "digest": "sha256:" + "7" * 64,
            "size_bytes": 1,
        },
    )

    result = worker.advance_configured_controls_plan(
        plan_path=plan_path,
        launch_state_root=tmp_path / "unused-launch-runs",
        progression_root=tmp_path / "progressions",
        preparation_queue_root=prep_root,
        activation_queue_root=tmp_path / "activations",
        publisher_factory=lambda: object(),
    )

    assert result["status"] == "construction_activation_queued"
    assert (scoped / "construction_activation_progression.json").is_file()
    assert not (launch / "construction_activation_progression.json").is_file()


def test_paid_transition_uses_only_injected_webapp_submitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authority = tmp_path / "inputs" / "construction-launch-authority.json"
    _write(authority, {"path": authority.name})
    plan["phases"]["construction"]["launch_authority_path"] = str(authority)
    _seal_plan(plan)
    _write(plan_path, plan)
    state = (
        tmp_path
        / "progressions"
        / plan["source_launch_id"]
        / f"franka-controls-{str(plan['expected_production_commit'])[:12]}"
    )
    _write(
        state / "configured_controls_progression.v1.json",
        _sealed_progression(
            "episode_preparation_queued",
            episode_preparation_request={"preparation_id": "prep-1"},
        ),
    )
    construction = _sealed_progression("construction_activation_queued")
    _write(state / "construction_activation_progression.json", construction)
    activation_id = plan["future_outputs"]["construction"][
        "expected_activation_id"
    ]
    profile_id = "construction-profile"
    profile_digest = "sha256:" + "9" * 64
    _write(
        tmp_path / "activations" / "results" / f"{activation_id}-digest.json",
        {
            "activation_id": activation_id,
            "profile_id": profile_id,
            "profile_digest": profile_digest,
        },
    )
    _write(
        tmp_path / "profiles" / f"{profile_id}.json",
        {"profile_id": profile_id, "profile_digest": profile_digest},
    )
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


def _construction_run(tmp_path: Path) -> tuple[Path, str, Path]:
    launch_id = "scene-839873-franka-construction-launch"
    run_root = tmp_path / "launch-runs" / launch_id
    authority_path = tmp_path / "inputs" / "attempt-authority.json"
    reconciliation_path = tmp_path / "inputs" / "spend-reconciliation.json"
    _write(
        authority_path,
        {"schema_version": "native_task_arena_paid_attempt_authority.v1"},
    )
    _write(
        reconciliation_path,
        {"schema_version": "adp_same_goal_spend_reconciliation.v1"},
    )
    construction_path = run_root / "allocator" / "construction-result.json"
    construction: dict[str, object] = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "candidate_policy_queried": False,
        "blockers": [],
        "result_digest": "",
    }
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    _write(construction_path, construction)
    allocator_path = run_root / "allocator" / "result.json"
    _write(
        allocator_path,
        {
            "schema_version": "native_task_arena_vast_run.v1",
            "status": "completed",
            "blockers": [],
            "native_control_result_path": str(construction_path),
            "native_control_result_digest": construction["result_digest"],
        },
    )
    profile: dict[str, object] = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": "scene-839873-franka-construction",
        "immutable_inputs": [
            {
                "name": "native_task_arena_attempt_authority",
                "path": str(authority_path),
                "digest": _digest(authority_path),
            },
            {
                "name": (
                    "native_task_arena_attempt_authority_"
                    "prior_spend_reconciliation"
                ),
                "path": str(reconciliation_path),
                "digest": _digest(reconciliation_path),
            },
        ],
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    _write(run_root / "launch_profile.json", profile)
    receipt: dict[str, object] = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "completed",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": "sha256:" + "1" * 64,
        "launch_profile_digest": profile["profile_digest"],
        "terminal_evidence": {
            "status": "passed",
            "result": {
                "path": str(allocator_path),
                "exists": True,
                "digest": _digest(allocator_path),
            },
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(run_root / "launch_receipt.json", receipt)
    sync: dict[str, object] = {
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        "launch_id": launch_id,
        "run_id": launch_id,
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
            "run_id": launch_id,
            "request_digest": receipt["request_digest"],
            "receipt_digest": receipt["receipt_digest"],
        },
        "sync_result_digest": "",
    }
    sync["sync_result_digest"] = canonical_digest(
        sync, digest_field="sync_result_digest"
    )
    _write(run_root / "webapp_sync_succeeded.json", sync)
    zero: dict[str, object] = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "launch_id": launch_id,
        "run_id": launch_id,
        "request_digest": receipt["request_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "launch_profile_digest": profile["profile_digest"],
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
    return run_root.parent, launch_id, construction_path


def test_construction_predecessor_is_derived_only_after_terminal_provider_zero(
    tmp_path: Path,
) -> None:
    launch_root, launch_id, _ = _construction_run(tmp_path)

    def publish(*, path: Path, object_name: str) -> dict[str, object]:
        size = path.stat().st_size
        digest = _digest(path)
        return {
            "uri": f"s3://test/{object_name}",
            "digest": digest,
            "size_bytes": size,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": digest,
            "readback_size_bytes": size,
        }

    predecessor = worker._construction_predecessor(
        launch_state_root=launch_root,
        construction_launch_id=launch_id,
        publisher=publish,
    )

    assert predecessor is not None
    lineage, paths = predecessor
    assert lineage["kind"] == "predecessor"
    assert set(paths) == {
        "prior_authority",
        "prior_result",
        "prior_launch_receipt",
        "prior_webapp_sync",
        "prior_provider_zero",
        "prior_spend_reconciliation",
        "construction_result",
    }
    assert all(lineage[name]["digest"] == _digest(Path(path)) for name, path in paths.items())


def test_construction_predecessor_refuses_changed_qualified_result(
    tmp_path: Path,
) -> None:
    launch_root, launch_id, construction_path = _construction_run(tmp_path)
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    construction["construction_gate_qualified"] = False
    _write(construction_path, construction)

    with pytest.raises(
        worker.TaskEvaluationConfiguredControlsProgressionWorkerError,
        match="configured_controls_worker_construction_result_invalid",
    ):
        worker._construction_predecessor(
            launch_state_root=launch_root,
            construction_launch_id=launch_id,
            publisher=lambda **_: {},
        )


def test_construction_predecessor_refuses_self_sealed_result_substitution(
    tmp_path: Path,
) -> None:
    launch_root, launch_id, construction_path = _construction_run(tmp_path)
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    construction["scene_plan_digest"] = "sha256:" + "9" * 64
    construction["result_digest"] = canonical_digest(
        construction, digest_field="result_digest"
    )
    _write(construction_path, construction)

    with pytest.raises(
        worker.TaskEvaluationConfiguredControlsProgressionWorkerError,
        match="configured_controls_worker_construction_result_invalid",
    ):
        worker._construction_predecessor(
            launch_state_root=launch_root,
            construction_launch_id=launch_id,
            publisher=lambda **_: {},
        )


def test_timer_derives_controls_lineage_after_construction_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    state = (
        tmp_path
        / "progressions"
        / plan["source_launch_id"]
        / f"franka-controls-{str(plan['expected_production_commit'])[:12]}"
    )
    _write(
        state / "configured_controls_progression.v1.json",
        _sealed_progression(
            "episode_preparation_queued",
            episode_preparation_request={"preparation_id": "prep-1"},
        ),
    )
    _write(
        state / "construction_activation_progression.json",
        _sealed_progression("construction_activation_queued"),
    )
    _write(
        state / "construction_launch_progression.json",
        _sealed_progression(
            "construction_launch_queued", launch_id="construction-launch-1"
        ),
    )
    prep_root = tmp_path / "preparations"
    _write(prep_root / "identities" / "prep-1.json", {"identity": "prep-1"})
    _write(prep_root / "results" / "prep-1-a.json", {"status": "prepared"})
    artifact_names = (
        "prior_authority",
        "prior_result",
        "prior_launch_receipt",
        "prior_webapp_sync",
        "prior_provider_zero",
        "prior_spend_reconciliation",
        "construction_result",
    )
    paths = {
        name: tmp_path / "derived-lineage" / f"{name}.json"
        for name in artifact_names
    }
    for path in paths.values():
        _write(path, {"schema_version": "test.derived.v1"})
    lineage = {"kind": "predecessor"}
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        worker,
        "_construction_predecessor",
        lambda **_: (lineage, {name: str(path) for name, path in paths.items()}),
    )

    def stage(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return _sealed_progression("controls_activation_queued")

    monkeypatch.setattr(worker, "stage_configured_controls_activation", stage)
    monkeypatch.setattr(
        worker,
        "_materialize_phase_release_window",
        lambda **_: {
            "uri": "s3://configured-controls/window.json",
            "digest": "sha256:" + "7" * 64,
            "size_bytes": 1,
        },
    )
    result = worker.advance_configured_controls_plan(
        plan_path=plan_path,
        launch_state_root=tmp_path / "launch-runs",
        progression_root=tmp_path / "progressions",
        preparation_queue_root=prep_root,
        activation_queue_root=tmp_path / "activations",
        publisher_factory=lambda: object(),
    )

    assert result["status"] == "controls_activation_queued"
    assert observed["lineage"] == lineage
    assert observed["lineage_artifact_paths"] == {
        name: str(path) for name, path in paths.items()
    }
    assert "lineage_path" not in plan["phases"]["controls"]


def test_systemd_worker_is_separate_from_reconciler_and_never_calls_allocator() -> None:
    service = Path(
        "deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service"
    ).read_text(encoding="utf-8")
    reconciler = Path(
        "deploy/systemd/blueprint-task-evaluation-launch-reconciler.service"
    ).read_text(encoding="utf-8")
    assert "task_evaluation_configured_controls_progression_worker" in service
    assert "User=blueprint" in service
    assert "Group=blueprint" in service
    assert "WorkingDirectory=/root" not in service
    assert "/root/" not in service
    assert (
        "Environment=OPENAI_API_KEY_FILE=/etc/blueprint/provider-secrets/"
        "openai_api_key_artifixer_visual_review"
    ) in service
    assert (
        "Environment=OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE="
        "/etc/blueprint/provider-secrets/openai_api_key_artifixer_visual_review"
    ) in service
    assert (
        'OPENAI_API_KEY_FILE="$${OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE}"'
        in service
    )
    assert "Environment=BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true" in service
    assert (
        "Environment=VAST_LAUNCH_LOCK_FILE=/var/lib/blueprint/"
        "pipeline-control-plane/provider-locks/vast_paid_launch.lock"
    ) in service
    assert "post_teardown" not in reconciler
    assert "paid_resource_allocator" not in service
    assert "vast_provider_adapter" not in service
    assert "--webapp-secret-file" in service
    assert "BLUEPRINT_TASK_EVALUATION_CONFIGURED_CONTROLS_PLAN_ROOT" in service
    assert (
        "BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_SECRET_FILE="
        "/etc/blueprint/provider-secrets/"
        "blueprint_task_evaluation_launch_submit_secret"
    ) in service


def test_installer_provisions_required_plan_root_and_canonical_secret() -> None:
    installer = Path("scripts/install_live_pipeline_control_plane.sh").read_text(
        encoding="utf-8"
    )
    assert "task-evaluation-configured-controls-plans" in installer
    assert "blueprint_task_evaluation_launch_submit_secret" in installer
    assert 'PLAN_OWNER="$(stat' in installer
    assert 'SECRET_OWNER="$(stat' in installer
    assert 'chmod 0750 "${CONFIGURED_CONTROLS_PLAN_ROOT}"' in installer
    assert 'chmod 0440 "${CONFIGURED_CONTROLS_WEBAPP_SECRET}"' in installer


def test_default_readiness_publication_uses_preparation_admitted_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def factory(**kwargs: object) -> object:
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(worker, "configured_scene_object_store_publisher", factory)
    worker.configured_controls_object_store_publisher()
    assert observed == {
        "key_prefix": "task-evaluation/production-inputs/configured-controls"
    }
